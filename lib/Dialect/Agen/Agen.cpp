//===-- Agen.cpp -------------------------------------------------*- c++ -*-==//
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
// This file implements the agen dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/Agen.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>

#include <cassert>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"

using namespace mlir;
using namespace mlir::agen;

//===----------------------------------------------------------------------===//
// Agen Enums
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/AgenEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom Parsers and Printers
//===----------------------------------------------------------------------===//

namespace {

auto parseAffineMapOfSSAIds(
    OpAsmParser& parser, AffineMapAttr& map,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) -> ParseResult {
  // MLIR's parseAffineMapOfSSAIds always builds a NamedAttribute from attrName
  // before returning, and NamedAttribute requires a non-empty name. We return
  // the map via the `map` out-param and discard the named attribute, but the
  // name must still be valid, so pass a placeholder rather than "".
  NamedAttrList ignored;
  return parser.parseAffineMapOfSSAIds(operands, map, "map", ignored,
                                       AsmParser::Delimiter::Square);
}

void printAffineMapOfSSAIds(OpAsmPrinter& printer, Operation* /*op*/,
                            AffineMapAttr map, ValueRange operands) {
  printer << "[";
  printer.printAffineMapOfSSAIds(map, operands);
  printer << "]";
}

}  // namespace

//===----------------------------------------------------------------------===//
// AgenDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/AgenDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/Agen/AgenAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Agen/Agen.cpp.inc"

void AgenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Agen/Agen.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/Agen/AgenAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// VectorStoreOp
//===----------------------------------------------------------------------===//

// Use identity map.
void VectorStoreOp::build(OpBuilder& builder, OperationState& state,
                          Value value_to_store, Value mem_ref,
                          /*optional*/ StringAttr dbg_name, ValueRange indices,
                          IntegerSet store_set, AffineMap store_order) {
  const auto memref_type = cast<MemRefType>(mem_ref.getType());
  build(builder, state, value_to_store, mem_ref, dbg_name,
        builder.getMultiDimIdentityMap(memref_type.getRank()), indices,
        store_set, store_order);
}

//===----------------------------------------------------------------------===//
// CompositeLoadAndStoreOp
//===----------------------------------------------------------------------===//

ParseResult CompositeLoadAndStoreOp::parse(OpAsmParser& parser,
                                           OperationState& result) {
  OpAsmParser::UnresolvedOperand src_mem_ref;
  AffineMapAttr src_map;
  SmallVector<OpAsmParser::UnresolvedOperand> src_operands;
  if (parser.parseKeyword("src") || parser.parseColon() ||
      parser.parseOperand(src_mem_ref) ||
      parseAffineMapOfSSAIds(parser, src_map, src_operands)) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand dst_mem_ref;
  AffineMapAttr dst_map;
  SmallVector<OpAsmParser::UnresolvedOperand> dst_operands;
  if (parser.parseKeyword("dst") || parser.parseColon() ||
      parser.parseOperand(dst_mem_ref) ||
      parseAffineMapOfSSAIds(parser, dst_map, dst_operands)) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> time_symbols;
  if (parser.parseKeyword("time_symbols") ||
      parser.parseOperandList(time_symbols, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }

  OpAsmParser::Argument load_induction_var;
  if (parser.parseComma() || parser.parseKeyword("load_iv") ||
      parser.parseLParen() || parser.parseArgument(load_induction_var, true) ||
      parser.parseRParen()) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand multicast_info;
  if (!parser.parseOptionalComma() &&
      (parser.parseKeyword("multicast_info") || parser.parseEqual() ||
       parser.parseOperand(multicast_info))) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.parseRegion(*result.addRegion(), {load_induction_var})) {
    return failure();
  }

  Type src_mem_ref_type;
  Type dst_mem_ref_type;
  if (parser.parseColon() || parser.parseType(src_mem_ref_type) ||
      parser.parseComma() || parser.parseType(dst_mem_ref_type)) {
    return failure();
  }

  const auto index_type = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(src_mem_ref, src_mem_ref_type, result.operands) ||
      parser.resolveOperand(dst_mem_ref, dst_mem_ref_type, result.operands) ||
      parser.resolveOperands(src_operands, index_type, result.operands) ||
      parser.resolveOperands(dst_operands, index_type, result.operands) ||
      parser.resolveOperands(time_symbols, index_type, result.operands)) {
    return failure();
  }
  if (multicast_info.location.isValid() &&
      parser.resolveOperand(multicast_info, index_type, result.operands)) {
    return failure();
  }

  result.addAttribute(getSrcMapAttrName(result.name), src_map);
  result.addAttribute(getDstMapAttrName(result.name), dst_map);
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, static_cast<int32_t>(src_operands.size()),
                           static_cast<int32_t>(dst_operands.size()),
                           static_cast<int32_t>(time_symbols.size()),
                           multicast_info.location.isValid() ? 1 : 0}));
  return success();
}

void CompositeLoadAndStoreOp::print(OpAsmPrinter& p) {
  p << " src:" << getSrcMemRef();
  printAffineMapOfSSAIds(p, *this, getSrcMapAttr(), getSrcOperands());
  p << " dst:" << getDstMemRef();
  printAffineMapOfSSAIds(p, *this, getDstMapAttr(), getDstOperands());

  p.printNewline();
  p << " time_symbols(";
  p.printOperands(getTimeSymbols());
  p << "), load_iv(";
  p << getLoadInductionVar() << ":" << getLoadInductionVar().getType();
  p << ')';

  if (const auto multicast_info = getMulticastInfo(); multicast_info) {
    p << ", multicast_info = " << multicast_info;
  }

  p.printNewline();
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSrcMapAttrName(), getDstMapAttrName(),
                       getOperandSegmentSizesAttrName()});

  p.printNewline();
  p.printRegion(getBodyRegion(), false, true);
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType();
}

void CompositeLoadAndStoreOp::build(
    OpBuilder& builder, OperationState& state, Value src_mem_ref,
    Value dst_mem_ref, StringAttr dbg_name, AffineMapAttr src_map,
    ValueRange src_operands, AffineMapAttr dst_map, ValueRange dst_operands,
    IntegerSetAttr load_set, AffineMapAttr load_order, IntegerSetAttr store_set,
    AffineMapAttr store_order, ValueRange time_symbols, IntegerSetAttr time_set,
    AffineMapAttr time_order, AffineMapAttr load_time_addr_map,
    AffineMapAttr store_time_addr_map, VectorType type,
    AgenRoutingDirectionAttr dir, Value multicast_info,
    function_ref<void(OpBuilder&, Location, Value)> body_builder) {
  state.addOperands({src_mem_ref, dst_mem_ref});
  state.addOperands(src_operands);
  state.addOperands(dst_operands);
  state.addOperands(time_symbols);
  if (multicast_info) {
    state.operands.push_back(multicast_info);
  }

  state.addAttribute(getSrcMapAttrName(state.name), src_map);
  state.addAttribute(getDstMapAttrName(state.name), dst_map);
  state.addAttribute(getLoadSetAttrName(state.name), load_set);
  state.addAttribute(getLoadOrderAttrName(state.name), load_order);
  state.addAttribute(getStoreSetAttrName(state.name), store_set);
  state.addAttribute(getStoreOrderAttrName(state.name), store_order);
  state.addAttribute(getTimeSetAttrName(state.name), time_set);
  state.addAttribute(getTimeOrderAttrName(state.name), time_order);
  state.addAttribute(getLoadTimeAddrMapAttrName(state.name),
                     load_time_addr_map);
  state.addAttribute(getStoreTimeAddrMapAttrName(state.name),
                     store_time_addr_map);
  if (dbg_name) {
    state.addAttribute(getDbgNameAttrName(state.name), dbg_name);
  }
  if (dir) {
    state.addAttribute(getDirAttrName(state.name), dir);
  }

  state.addAttribute(
      getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr(
          {1, 1, static_cast<int32_t>(src_operands.size()),
           static_cast<int32_t>(dst_operands.size()),
           static_cast<int32_t>(time_symbols.size()), multicast_info ? 1 : 0}));

  auto* body = &state.addRegion()->emplaceBlock();
  const auto load_induction_var = body->addArgument(type, state.location);
  ensureTerminator(*state.regions.front(), builder, state.location);

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    body_builder(builder, state.location, load_induction_var);
  }
}

void CompositeLoadAndStoreOp::build(
    OpBuilder& builder, OperationState& state, Value src_mem_ref,
    Value dst_mem_ref, StringAttr dbg_name, AffineMap src_map,
    ValueRange src_operands, AffineMap dst_map, ValueRange dst_operands,
    IntegerSet load_set, AffineMap load_order, IntegerSet store_set,
    AffineMap store_order, ValueRange time_symbols, IntegerSet time_set,
    AffineMap time_order, AffineMap load_time_addr_map,
    AffineMap store_time_addr_map, VectorType type,
    AgenRoutingDirectionAttr dir, Value multicast_info,
    function_ref<void(OpBuilder&, Location, Value)> body_builder) {
  build(builder, state, src_mem_ref, dst_mem_ref, dbg_name,
        AffineMapAttr::get(src_map), src_operands, AffineMapAttr::get(dst_map),
        dst_operands, IntegerSetAttr::get(load_set),
        AffineMapAttr::get(load_order), IntegerSetAttr::get(store_set),
        AffineMapAttr::get(store_order), time_symbols,
        IntegerSetAttr::get(time_set), AffineMapAttr::get(time_order),
        AffineMapAttr::get(load_time_addr_map),
        AffineMapAttr::get(store_time_addr_map), type, dir, multicast_info,
        body_builder);
}

auto CompositeLoadAndStoreOp::verify() -> LogicalResult {
  if (getLoadSet().getValue().getNumDims() < getSrcMap().getNumResults()) {
    return emitOpError() << "load set and src array dimensions should match";
  }

  if (getLoadSet().getValue().getNumDims() != getLoadOrder().getNumDims()) {
    return emitOpError() << "load set and order dimensions should match";
  }

  if (getStoreSet().getValue().getNumDims() < getDstMap().getNumResults()) {
    return emitOpError() << "store set and dst array dimensions should match";
  }

  if (getStoreSet().getValue().getNumDims() != getStoreOrder().getNumDims()) {
    return emitOpError() << "store set and order dimensions should match";
  }

  return success();
}

auto CompositeLoadAndStoreOp::getAffineMapAttrForMemRef(Value memref)
    -> NamedAttribute {
  if (memref == getSrcMemRef()) {
    return {getSrcMapAttrName(), getSrcMapAttr()};
  }

  if (memref == getDstMemRef()) {
    return {getDstMapAttrName(), getDstMapAttr()};
  }

  llvm_unreachable("invalid operand value");
}
