//===-- KTDFArchAttributes.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// I64Attr
//===----------------------------------------------------------------------===//

auto I64Attr::get(MLIRContext* context, int64_t value) -> I64Attr {
  return cast<I64Attr>(IntegerAttr::get(IntegerType::get(context, 64U), value));
}

//===----------------------------------------------------------------------===//
// AdjacencyMatrixAttr
//===----------------------------------------------------------------------===//

auto AdjacencyMatrixAttr::get(MLIRContext* context, int64_t dim,
                              ArrayRef<bool> values) -> AdjacencyMatrixAttr {
  assert(dim >= 0);

  const auto i1_type = IntegerType::get(context, 1U);
  const auto type = RankedTensorType::get({dim, dim}, i1_type);

  return cast<AdjacencyMatrixAttr>(DenseIntElementsAttr::get(type, values));
}

auto AdjacencyMatrixAttr::get(MLIRContext* context, int64_t dim,
                              ArrayRef<Edge> adjacencies)
    -> AdjacencyMatrixAttr {
  assert(dim >= 0);

  const auto i1_type = IntegerType::get(context, 1U);
  const auto type = RankedTensorType::get({dim, dim}, i1_type);

  const auto num_indices = static_cast<int64_t>(adjacencies.size());
  const auto indices = DenseIntElementsAttr::get(
      RankedTensorType::get({num_indices, 2}, IntegerType::get(context, 64U)),
      ArrayRef<int64_t>(reinterpret_cast<const int64_t*>(adjacencies.data()),
                        num_indices * 2));
  const auto values = DenseIntElementsAttr::get(
      RankedTensorType::get({num_indices}, i1_type), true);

  return cast<AdjacencyMatrixAttr>(
      SparseElementsAttr::get(type, indices, values));
}

auto AdjacencyMatrixAttr::contains(int64_t source, int64_t target) const
    -> bool {
  assert(source >= 0 && target >= 0);
  const auto dim = getDim();
  assert(source < dim && target < dim);

  return getValues<bool>()[(source * dim) + target];
}

//===----------------------------------------------------------------------===//
// KTDFArchDialect
//===----------------------------------------------------------------------===//

void KTDFArchDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// MapAttr
//===----------------------------------------------------------------------===//

auto MapAttr::parse(AsmParser& parser, Type /*type*/) -> Attribute {
  if (parser.parseLess()) {
    return {};
  }

  SmallVector<Entry> entries;
  if (parser.parseOptionalGreater()) {
    llvm::SmallDenseSet<Attribute> keys;
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          const auto key_loc = parser.getCurrentLocation();
          Attribute key;
          if (parser.parseAttribute(key) || parser.parseEqual()) {
            return failure();
          }
          if (!keys.insert(key).second) {
            return parser.emitError(key_loc)
                   << "multiple entries for key " << key;
          }

          Attribute value;
          if (parser.parseAttribute(value)) {
            return failure();
          }

          entries.emplace_back(key, value);
          return success();
        })) {
      return {};
    }
  }

  return MapAttr::get(parser.getContext(), entries);
}

void MapAttr::print(AsmPrinter& printer) const {
  printer << "<";
  llvm::interleaveComma(getEntries(), printer, [&](auto& entry) {
    printer << entry.first << " = " << entry.second;
  });
  printer << ">";
}

auto MapAttr::verify(function_ref<InFlightDiagnostic()> emit_error,
                     ArrayRef<Entry> entries) -> LogicalResult {
  llvm::SmallDenseSet<Attribute> keys;
  for (auto [key, _] : entries) {
    if (!keys.insert(key).second) {
      return emit_error() << "multiple entries for key " << key;
    }
  }

  return success();
}
