//===-- KTDFAttributes.cpp --------------------------------------*- c++ -*-===//
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
// This file implements the attributes in the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// KTDFDialect
//===----------------------------------------------------------------------===//

void KTDFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// UniqueProp
//===----------------------------------------------------------------------===//

auto UniqueProp::asAttr(MLIRContext* ctx) const -> Attribute {
  return IntegerAttr::get(IndexType::get(ctx), identity_);
}

auto UniqueProp::setFromAttr(UniqueProp& prop, Attribute attr,
                             function_ref<InFlightDiagnostic()> diag)
    -> LogicalResult {
  const auto int_attr = dyn_cast<IntegerAttr>(attr);
  if (!int_attr || !isa<IndexType>(int_attr.getType())) {
    return diag() << "expected index attribute";
  }

  const auto maybe_value = int_attr.getValue().tryZExtValue();
  if (!maybe_value) {
    return diag() << "expected host word sized integer";
  }
  prop.identity_ = *maybe_value;
  return success();
}

void UniqueProp::write(DialectBytecodeWriter& writer) const {
  writer.writeVarInt(identity_);
}

auto UniqueProp::read(DialectBytecodeReader& reader, UniqueProp& result)
    -> LogicalResult {
  return reader.readVarInt(result.identity_);
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.cpp.inc"
