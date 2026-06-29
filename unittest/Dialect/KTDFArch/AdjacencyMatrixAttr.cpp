//===----------------------------------------------------------------------===//
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

#include <doctest/doctest.h>
#include <llvm/ADT/Sequence.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

TEST_CASE("mlir::ktdf_arch::AdjacencyMatrixAttr") {
  MLIRContext context;

  SUBCASE("get splat") {
    const int64_t dim = 4;
    const auto splat = AdjacencyMatrixAttr::get(&context, dim, true);

    CHECK(isa<DenseIntElementsAttr>(splat));
    CHECK_EQ(splat.getElementType(), IntegerType::get(&context, 1));
    CHECK_EQ(splat.getNumElements(), dim * dim);

    for (auto [from, to] :
         llvm::zip_equal(llvm::iota_range<int64_t>(0, dim, false),
                         llvm::iota_range<int64_t>(0, dim, false))) {
      CHECK(splat.contains(from, to));
    }
  }

  SUBCASE("get dense") {
    // clang-format off
    const auto dense = AdjacencyMatrixAttr::get(&context, 3, {
      true, true, true,
      false, true, true,
      false, false, true,
    });
    // clang-format on

    CHECK(isa<DenseIntElementsAttr>(dense));
    CHECK_EQ(dense.getElementType(), IntegerType::get(&context, 1));
    CHECK_EQ(dense.getNumElements(), 9);

    for (auto [from, to] :
         llvm::zip_equal(llvm::iota_range<int64_t>(0, 3, false),
                         llvm::iota_range<int64_t>(0, 3, false))) {
      CHECK_EQ(dense.contains(from, to), to >= from);
    }
  }

  SUBCASE("get sparse") {
    // clang-format off
    const auto sparse = AdjacencyMatrixAttr::get(&context, 3, {
      AdjacencyMatrixAttr::Edge{0, 0},
      AdjacencyMatrixAttr::Edge{0, 1},
      AdjacencyMatrixAttr::Edge{1, 1},
      AdjacencyMatrixAttr::Edge{2, 0}
    });
    // clang-format on

    CHECK(isa<SparseElementsAttr>(sparse));
    CHECK_EQ(sparse.getElementType(), IntegerType::get(&context, 1));
    CHECK_EQ(sparse.getNumElements(), 9);

    // clang-format off
    CHECK_EQ(
      llvm::to_vector(sparse.getValues<bool>()),
      ArrayRef<bool>{ 
        true, true, false,
        false, true, false,
        true, false, false
      });
    //clang-format on
  }
}
