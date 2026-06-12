//===-- KTDFAttributes.h ----------------------------------------*- c++ -*-===//
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
// This file defines the attributes in the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFATTRIBUTES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFATTRIBUTES_H_

#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>

#include <cstdint>

#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"    // IWYU pragma: keep

namespace mlir {

class DialectBytecodeWriter;
class DialectBytecodeReader;

}  // namespace mlir

namespace mlir::ktdf {

struct UniqueProp {
  /*implicit*/ UniqueProp() {
    identity_ = reinterpret_cast<std::uintptr_t>(this);
  }
  ~UniqueProp() = default;
  /*implicit*/ UniqueProp(UniqueProp&&) = default;
  UniqueProp& operator=(UniqueProp&&) = default;
  /*implicit*/ UniqueProp(const UniqueProp&) = default;
  UniqueProp& operator=(const UniqueProp&) = default;

  auto operator==(const UniqueProp& rhs) const {
    return identity_ == rhs.identity_;
  }
  auto operator!=(const UniqueProp& rhs) const { return !(*this == rhs); }

  auto asAttr(MLIRContext* ctx) const -> Attribute;
  static auto setFromAttr(UniqueProp& prop, Attribute attr,
                          function_ref<InFlightDiagnostic()> diag)
      -> LogicalResult;

  void write(DialectBytecodeWriter& writer) const;
  static auto read(DialectBytecodeReader& reader, UniqueProp& result)
      -> LogicalResult;

 private:
  friend auto hash_value(const mlir::ktdf::UniqueProp& prop)
      -> llvm::hash_code {
    return prop.identity_;
  }

  std::uint64_t identity_;
};

}  // namespace mlir::ktdf

/// Auto-generated includes.
#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h.inc"

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFATTRIBUTES_H_
