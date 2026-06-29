//===-- Utils.h --------------------------------------------------*- c++ -*-==//
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
// This file declares some utilities for use with the agen dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_AGEN_UTILS_H_
#define DATAFLOW_SCHEDULER_DIALECT_AGEN_UTILS_H_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir {

class AffineMap;

}  // namespace mlir

namespace mlir::agen {

/// @brief Construct a map of iterator coefficients from loop iterators to
/// their constant coefficients calculated using the layout map.
/// @param subscripts_map The subscripts map to calculate coefficients for.
/// @param transfer_order A map describing how the subscripts map should be
/// ordered.
/// @param mem_view_layout_map The 1D map describing how to stride through
/// memory dimensions.
/// @param indices SmallVector<Value> containing the Values representing the
/// subscripts map dimensions.
/// @return DenseMap<Value, int64_t> containing the indices to composed
/// coefficients.
DenseMap<Value, int64_t> constructIteratorCoeffDict(
    const AffineMap& subscripts_map, const AffineMap& transfer_order,
    const AffineMap& mem_view_layout_map, SmallVectorImpl<Value>& indices);

}  // namespace mlir::agen

#endif  // DATAFLOW_SCHEDULER_DIALECT_AGEN_UTILS_H_

// Made with Bob
