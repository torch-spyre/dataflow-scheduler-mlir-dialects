//===-- KTDFTypes.h ---------------------------------------------*- c++ -*-===//
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
// This file defines the types in the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFTYPES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFTYPES_H_

#include <mlir/IR/TypeUtilities.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"  // IWYU pragma: keep

/// Auto-generated includes.
#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h.inc"

namespace mlir::ktdf {

/// @copydoc getElementTypeOrSelf(Type)
inline auto getElementTypeOrSelf(FifoSlotType type) -> Type {
  return type.getElementType();
}

/// Obtains the element type of @p type, or the type itself.
///
/// This is an extension of mlir::getElementTypeOrSelf that also supports the
/// ktdf::FifoSlotType, falling back to the MLIR implementation otherwise.
auto getElementTypeOrSelf(Type type) -> Type;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFTYPES_H_
