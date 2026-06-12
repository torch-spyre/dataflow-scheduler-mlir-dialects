//===-- KTDFTypes.cpp -------------------------------------------*- c++ -*-===//
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
// This file implements the types in the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::ktdf;

auto mlir::ktdf::getElementTypeOrSelf(Type type) -> Type {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case([](FifoSlotType type) { return getElementTypeOrSelf(type); })
      .Default([](Type type) { return mlir::getElementTypeOrSelf(type); });
}

//===----------------------------------------------------------------------===//
// KTDFDialect
//===----------------------------------------------------------------------===//

void KTDFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// FifoSlotType
//===----------------------------------------------------------------------===//

void FifoSlotType::print(AsmPrinter& printer) const {
  printer << "<";
  printer.printString(getFifoType().getValue());
  printer << ", ";
  if (ShapedType::isDynamic(getNumElements())) {
    printer << "?";
  } else {
    printer << getNumElements();
  }
  printer << "x";
  printer.printType(getElementType());
  printer << ">";
}

Type FifoSlotType::parse(AsmParser& parser) {
  if (parser.parseLess()) return Type();

  // Parse the FIFO connection type as a string keyword.
  std::string fifo_type_str;
  if (parser.parseKeywordOrString(&fifo_type_str)) return Type();

  if (parser.parseComma()) return Type();

  // Parse dimension (NxType or ?xType)
  int64_t num_elements;
  Type element_type;

  // Try to parse '?' for dynamic dimension
  if (succeeded(parser.parseOptionalQuestion())) {
    num_elements = ShapedType::kDynamic;
  } else {
    // Parse static dimension
    if (parser.parseInteger(num_elements)) return Type();
  }

  // Parse 'x' - use parseXInDimensionList which handles 'x' in dimension
  // context
  if (parser.parseXInDimensionList()) return Type();

  // Parse element type
  if (parser.parseType(element_type)) return Type();

  if (parser.parseGreater()) return Type();

  return FifoSlotType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(),
      StringAttr::get(parser.getContext(), fifo_type_str), num_elements,
      element_type);
}
