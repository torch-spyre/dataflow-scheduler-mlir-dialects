//===-- KTDFDialect.cpp -----------------------------------------*- c++ -*-===//
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
// This file implements the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// KTDFDialect
//===----------------------------------------------------------------------===//

void KTDFDialect::initialize() {
  registerOps();
  registerTypes();
  registerAttributes();
}

// Materialize a constant attribute as an arith.constant op. This is needed
// so that BufferPhaseOp::fold() can fold an all-constant buffer_phase to
// index 0 and have the greedy rewriter insert the corresponding constant.
auto KTDFDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                      Type type, Location loc)
    -> Operation* {
  if (arith::ConstantOp::isBuildableWith(value, type)) {
    return arith::ConstantOp::create(builder, loc, type, cast<TypedAttr>(value));
  }
  return nullptr;
}
