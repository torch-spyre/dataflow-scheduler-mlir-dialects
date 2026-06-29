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
// This file implements the dataflow dialect utilities.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"

using namespace mlir;
using namespace mlir::dataflow;

int mlir::dataflow::getCoreletId(const GetUnitOp op) {
  int corelet = -1;
  if (!op) return corelet;
  auto corelet_attr = op->getAttrOfType<IntegerAttr>("corelet");
  if (corelet_attr) {
    corelet = corelet_attr.getInt();
  }
  return corelet;
}
