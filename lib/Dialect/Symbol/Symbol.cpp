//===-- Symbol.cpp -----------------------------------------------*- c++ -*-==//
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
// This file implements the symbol dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::symbol;

//===----------------------------------------------------------------------===//
// SymbolDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Symbol/SymbolDialect.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Symbol/Symbol.cpp.inc"

void SymbolDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Symbol/Symbol.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SymbolImmutableMappingOp
//===----------------------------------------------------------------------===//

// syntax:%result = symbol.symbol_immutable_mapping([%1 -> %2],[%3 -> %4]):index
ParseResult SymbolImmutableMappingOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  auto &builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> keys, values;
  OpAsmParser::UnresolvedOperand key, value;
  auto op_result = parser.parseLParen().failed();
  while (parser.parseOptionalLSquare().succeeded()) {
    op_result = op_result || parser.parseOperand(key) || parser.parseArrow() ||
                parser.parseOperand(value) || parser.parseRSquare();
    auto parse_result = parser.parseOptionalComma();
    keys.push_back(key);
    values.push_back(value);
  }
  Type result_type;
  op_result =
      op_result || parser.parseRParen() || parser.parseColonType(result_type);
  result.types.push_back(result_type);
  int size = keys.size();
  result.attributes.push_back(
      builder.getNamedAttr(getOperandSegmentSizesAttrName(OperationName(
                               getOperationName(), builder.getContext())),
                           builder.getDenseI32ArrayAttr({size, size})));
  op_result = op_result ||
              parser.resolveOperands(keys, index_type, result.operands) ||
              parser.resolveOperands(values, index_type, result.operands);

  return failure(op_result);
}

void SymbolImmutableMappingOp::print(OpAsmPrinter &p) {
  auto &op = *this;

  int size = op.getKeys().size();
  p << '(';
  for (int i = 0; i < size; i++) {
    p << '[';
    p.printOperand(op.getKeys()[i]);
    p << " -> ";
    p.printOperand(op.getValues()[i]);
    p << ']';
    if (i < size - 1) p << ", ";
  }
  p << "):";
  p.printType(op.getResult().getType());
}

LogicalResult SymbolImmutableMappingOp::verify() {
  auto &op = *this;
  if (op.getKeys().size() != op.getValues().size()) {
    op.emitError("Mismatching keys/values sizes in SymbolImmutableMapping\n");
    return failure();
  }

  for (auto key : op.getKeys()) {
    Operation *key_def = key.getDefiningOp();
    if (!key_def || !key_def->hasTrait<mlir::OpTrait::ConstantLike>()) {
      op.emitError() << "All keys expected to be constant operations\n";
      return failure();
    }
  }

  for (auto value : op.getValues()) {
    if (!isa<symbol::CreateSymbolOp>(value.getDefiningOp())) {
      op.emitError(
          "Values in SymbolImmutableMapping should be results of "
          "symbol.create_symbol\n");
      return failure();
    }
    OpBuilder builder(op);
    if (value.getType() != builder.getIndexType()) {
      op.emitError(
          "Values in SymbolImmutableMapping expected to have Index type\n");
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolQueryMapOp
//===----------------------------------------------------------------------===//

// syntax:%result = symbol.query_map(map:%2, key:%1) : index
ParseResult SymbolQueryMapOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  OpAsmParser::UnresolvedOperand map, key;
  Type result_type;

  auto op_result = parser.parseLParen() || parser.parseKeyword("map") ||
                   parser.parseColon() || parser.parseOperand(map) ||
                   parser.parseComma() || parser.parseKeyword("key") ||
                   parser.parseColon() || parser.parseOperand(key) ||
                   parser.parseRParen() || parser.parseColonType(result_type);
  result.types.push_back(result_type);
  op_result = op_result ||
              parser.resolveOperand(map, index_type, result.operands) ||
              parser.resolveOperand(key, index_type, result.operands);

  return failure(op_result);
}

void SymbolQueryMapOp::print(OpAsmPrinter &p) {
  auto &op = *this;

  p << "(map:";
  p.printOperand(op.getMap());
  p << ", key:";
  p.printOperand(op.getKey());
  p << ") : ";
  p.printType(op.getResult().getType());
}

LogicalResult SymbolQueryMapOp::verify() {
  auto &op = *this;
  // Key expected to be a loop iterator argument or a Value.
  if (dyn_cast<BlockArgument>(op.getKey())) {
    auto parent_op = op.getKey().getParentRegion()->getParentOp();
    // Accept any loop-like op (i.e. one implementing LoopLikeOpInterface).
    if (!isa<mlir::LoopLikeOpInterface>(parent_op)) {
      op.emitError("key's parentOp must be a ForOp.");
      return failure();
    }
  }
  if (!op.getMap().getDefiningOp<SymbolImmutableMappingOp>()) {
    op.emitError("map's defOp must be symbol_immutable_mapping op.");
    return failure();
  }
  return success();
}
