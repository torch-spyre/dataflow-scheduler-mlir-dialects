//===-- Utils.cpp ------------------------------------------------*- c++ -*-==//
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
// This file implements the agen dialect utilities.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/Utils.h"

#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>

using namespace mlir;
using namespace mlir::agen;

namespace {

/// @brief Given \p composed_map , construct the composed coefficients.
/// @param composed_map AffineMap representing an ordered subscripts map
/// composed with a layout map.
/// @return SmallVector<int64_t> containing the individual composed iterator
/// coefficients.
SmallVector<int64_t> constructDimCoefficients(AffineMap& composed_map) {
  composed_map = compressUnusedSymbols(composed_map);
  composed_map = simplifyAffineMap(composed_map);

  SmallVector<int64_t> composed_coeffs;
  affine::FlatAffineValueConstraints composed_flat_csts;

  assert(composed_map.getResults().size() == 1);
  AffineExpr expr = composed_map.getResult(0);

  auto flat_result = getFlattenedAffineExpr(
      expr, composed_map.getNumDims(), composed_map.getNumSymbols(),
      &composed_coeffs, &composed_flat_csts);

  assert(flat_result.succeeded());

  return composed_coeffs;
}

}  // namespace

DenseMap<Value, int64_t> mlir::agen::constructIteratorCoeffDict(
    const AffineMap& subscripts_map, const AffineMap& transfer_order,
    const AffineMap& mem_view_layout_map, SmallVectorImpl<Value>& indices) {
  auto composed_load_order_map = transfer_order.compose(subscripts_map);
  auto composed_layout_map =
      mem_view_layout_map.compose(composed_load_order_map);
  affine::fullyComposeAffineMapAndOperands(&composed_layout_map, &indices);
  auto composed_layout_coeffs = constructDimCoefficients(composed_layout_map);

  // Sort the indices from outermost to innnermost
  DenseMap<Value, int64_t> indices_coeff_dict;
  for (std::size_t i = 0; i < indices.size(); ++i)
    indices_coeff_dict[indices[i]] = composed_layout_coeffs[i];

  int const_value =
      composed_layout_coeffs.size() != composed_layout_map.getNumDims()
          ? composed_layout_coeffs.back()
          : 0;

  // indices_coeff_dict is a map from loop iterators to coefficients associated
  // with them. nullptr refers to constant offset
  // for, e.g., 2xi + 3xj + 10
  // coefficient with i is 2, j is 3, and nullptr is 10.
  indices_coeff_dict[nullptr] = const_value;

  return indices_coeff_dict;
}

// Made with Bob
