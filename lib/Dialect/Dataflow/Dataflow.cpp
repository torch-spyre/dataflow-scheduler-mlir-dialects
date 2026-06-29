//===-- Dataflow.cpp ---------------------------------------------*- c++ -*-==//
//
// Part of the Dataflow Scheduler MLIR Dialects project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the dataflow dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/IntegerSet.h>

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Dataflow Enums
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Dataflow/DataflowEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// DataflowDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Dataflow/DataflowDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/Dataflow/DataflowAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/Dataflow/DataflowAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.cpp.inc"
      >();
}

/*
 void GetUnitOp::print(OpAsmPrinter &printer) {
  auto op = this;
  printer << op->getName() << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  if (op->getNumResults() != 0) {
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type) { return type == resultType; })) {
      printer << resultType;
      return;
    }
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}*/

// We couldn't find a way to cleanly specify a variadic result with definitely
// at-least one entry. So, we used variadic in td definition and check
// for single entry in the verify function.
LogicalResult GetUnitOp::verify() {
  auto& op = *this;
  return op.getNumResults() >= 1 ? LogicalResult::success() : failure();
}

Value GetUnitOp::getUnit() {
  auto& op = *this;
  return op.getResult(0);
}

//===----------------------------------------------------------------------===//
// ProgramUnitOp
//===----------------------------------------------------------------------===//
/* Build method to construct ProgramUnit Op */
void ProgramUnitOp::build(OpBuilder& builder, OperationState& result,
                          ValueRange unitIds, StringAttr precision_attr,
                          ProgramUnitOp::BodyBuilderFn bodyBuilder) {
  result.addOperands(unitIds);
  //  result.addTypes(unitId.getType());

  if (!precision_attr.getValue().str().empty()) {
    result.addAttribute(dataflow::ProgramUnitOp::getPrecisionAttrStrName(),
                        precision_attr);
  }

  // Add a body region with block arguments as unwrapped async value operands.
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block& bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(unitIds[0].getType(), result.location);

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the execute op.
  if (!bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    dataflow::ReturnOp::create(builder, result.location, TypeRange());
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    // int tmp = bodyBlock.getNumArguments();
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
  }
}

void ProgramUnitOp::build(OpBuilder& builder, OperationState& result,
                          ValueRange unitIds,
                          ProgramUnitOp::BodyBuilderFn bodyBuilder) {
  build(builder, result, unitIds,
        StringAttr::get(builder.getContext(), StringRef("")), bodyBuilder);
}

ParseResult ProgramUnitOp::parse(OpAsmParser& parser, OperationState& result) {
  ::llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> unitsOperands;
  ::llvm::SMLoc unitsOperandsLoc;
  (void)unitsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<Region>, 2> fullRegions;
  OpAsmParser::Argument region_arg;
  region_arg.type = parser.getBuilder().getIndexType();

  unitsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalKeyword("iter_arg").succeeded()) {
    if (parser.parseColon()) return failure();
    if (parser.parseArgument(region_arg)) return failure();
    if (parser.parseArrow()) return failure();
    if (parser.parseOperandList(unitsOperands, AsmParser::Delimiter::Paren))
      return failure();
  } else {
    if (parser.parseOperandList(unitsOperands)) return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  if (parser.parseColon()) return failure();

  {
    std::unique_ptr<Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region, region_arg);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult)) return failure();
      fullRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<Region>();
        if (parser.parseRegion(*region)) return failure();
        fullRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto& region : fullRegions)
    ensureTerminator(*region, parser.getBuilder(), result.location);
  result.addRegions(fullRegions);
  Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(unitsOperands, odsBuildableType0, unitsOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void ProgramUnitOp::print(OpAsmPrinter& _odsPrinter) {
  _odsPrinter << ' ';
  auto arguments = getRegion().getArguments();
  if (arguments.size() > 0) {
    _odsPrinter << "iter_arg : ";
    _odsPrinter << arguments;
    _odsPrinter << " -> (";
  }
  _odsPrinter << getUnits();
  if (arguments.size() > 0) {
    _odsPrinter << ')';
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  llvm::interleaveComma(
      getOperation()->getRegions(), _odsPrinter, [&](Region& region) {
        {
          bool printTerminator = true;
          if (auto* term =
                  region.empty() ? nullptr : region.begin()->getTerminator()) {
            printTerminator = !term->getAttrDictionary().empty() ||
                              term->getNumOperands() != 0 ||
                              term->getNumResults() != 0;
          }
          _odsPrinter.printRegion(region, /*printEntryBlockArgs=*/false,
                                  /*printBlockTerminators=*/printTerminator);
        }
      });
}
