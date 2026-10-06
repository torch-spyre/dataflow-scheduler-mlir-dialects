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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"

#include <doctest/doctest.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

TEST_CASE("mlir::ktdf::PipelineBuilder") {
  MLIRContext context;
  context.loadDialect<KTDFDialect>();
  OpBuilder builder(&context);
  auto module =
      OwningOpRef<ModuleOp>(ModuleOp::create(builder.getUnknownLoc()));
  builder.setInsertionPointToEnd(module->getBody());

  // Building erases the empty PrivateOp, and the builders build again when
  // they are destroyed.
  SUBCASE("empty pipeline, destroyed") {
    {
      PipelineBuilder pipeline_builder(builder, builder.getUnknownLoc());
    }
    auto pipelines = module->getOps<PipelineOp>();
    REQUIRE_FALSE(pipelines.empty());
    CHECK_FALSE((*pipelines.begin()).getPrivateOp());
  }

  SUBCASE("empty pipeline, built then destroyed") {
    PipelineOp pipeline;
    {
      PipelineBuilder pipeline_builder(builder, builder.getUnknownLoc());
      pipeline = pipeline_builder.build();
      CHECK_FALSE(pipeline.getPrivateOp());
      CHECK(pipeline_builder.build() == pipeline);
    }
    CHECK_FALSE(pipeline.getPrivateOp());
  }
}
