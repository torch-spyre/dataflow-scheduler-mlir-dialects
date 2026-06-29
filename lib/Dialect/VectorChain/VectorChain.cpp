//===-- VectorChain.cpp ------------------------------------------*- c++ -*-==//
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
// This file implements the vectorchain dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::vectorchain;

//===----------------------------------------------------------------------===//
// VectorChain Enums
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/VectorChain/VectorChainEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// VectorChainDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/VectorChain/VectorChainDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/VectorChain/VectorChainAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.cpp.inc"

void VectorChainDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/VectorChain/VectorChainAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//
LogicalResult ShuffleOp::verify() {
  auto& op = *this;

  auto result_type = dyn_cast<VectorType>(op.getResult().getType());
  if (!result_type) {
    op->emitOpError("result type is not a vector");
    return failure();
  }

  // The number of elements in the result type should be equivalent to the
  // product of the number of elements in the indices and the number of
  // repetitions.
  auto indices = op.getIndices();
  auto indices_size = indices.size();
  auto result_size = static_cast<std::size_t>(result_type.getNumElements());
  if (result_size != indices_size * op.getRepetition()) {
    op->emitOpError("result does not scale with indices and repetitions");
    return failure();
  }

  auto input_type = op.getInput().getType();
  Type input_elements_type;
  int input_num_of_elements = 0;
  // The input to shuffle is expected to be a VectorTyep or IBMVectorFloatType.
  if (auto const_vtype = dyn_cast<VectorType>(input_type)) {
    input_elements_type = const_vtype.getElementType();
    input_num_of_elements = const_vtype.getNumElements();
  } else {
    op->emitOpError(
        "input to vectorchain.shuffle should be VectorType or "
        "IBMVectorFloatType.");
    return failure();
  }
  // The element type of the input should match the type of the output (type
  // only, not necessarily number of elements).
  if (result_type.getElementType() != input_elements_type) {
    op->emitOpError(
        "input element type does not match output element "
        "type");
    return failure();
  }
  // The elements referred to in the indices should not refer to out of bounds
  // elements from the input. Element values of -1 are still valid as these are
  // used to indicate padding.
  int max_element = input_num_of_elements - 1;
  for (std::size_t i = 0; i < indices_size; ++i) {
    auto element = dyn_cast<IntegerAttr>(indices[i]);
    if (!element) {
      op->emitOpError("indices element is not integer");
      return failure();
    }

    int64_t element_value = element.getInt();
    if (element_value < -1 || element_value > max_element) {
      op->emitOpError("out of bounds indices element");
      return failure();
    }
  }

  return success();
}
