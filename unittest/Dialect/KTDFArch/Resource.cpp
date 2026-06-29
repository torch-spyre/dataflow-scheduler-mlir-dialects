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
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

TEST_CASE("mlir::ktdf_arch::Resource") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "resource.mlir");
  auto device = *module->getOps<DeviceOp>().begin();
  auto datapath = *device.getBody()->getOps<DatapathOp>().begin();

  SUBCASE("getProperty<>()") {
    SUBCASE("absent") {
      const auto maps_to = datapath.getProperty<MapsToAttr>();
      REQUIRE_FALSE(maps_to);
    }

    SUBCASE("present") {
      const auto bandwidth = datapath.getProperty<BandwidthAttr>();
      REQUIRE(bandwidth);
      CHECK(bandwidth.getValue() == 128);
    }
  }

  SUBCASE("setProperty<>()") {
    datapath.setProperty(BandwidthAttr::get(&context, 42));
    const auto attr = datapath.getProperty<BandwidthAttr>();
    REQUIRE(attr);
    CHECK(attr.getValue() == 42);
  }

  SUBCASE("removeProperty<>()") {
    const auto removed = datapath.removeProperty<BandwidthAttr>();
    REQUIRE(removed);
    CHECK(removed.getValue() == 128);
    REQUIRE_FALSE(datapath.removeProperty<BandwidthAttr>());
  }

  SUBCASE("getFeature(StringRef)") {
    SUBCASE("absent") {
      auto feature = datapath.getFeature(feature::SIMD::getAttrName(&context));
      REQUIRE_FALSE(feature);
    }

    SUBCASE("present") {
      auto feature = datapath.getFeature(feature::Queue::getAttrName(&context));
      REQUIRE(feature);
    }
  }

  SUBCASE("getFeature<>()") {
    SUBCASE("absent") {
      auto feature = datapath.getFeature<feature::SIMD>();
      REQUIRE_FALSE(feature);
    }

    SUBCASE("present") {
      auto feature = datapath.getFeature<feature::Queue>();
      REQUIRE(feature);
      CHECK_EQ(feature.getSize(), 4096);
      CHECK_EQ(feature.getDepth(), 256);
      CHECK(feature.isOrdered());
    }

    SUBCASE("sufficient") {
      auto require = cast<feature::Queue>(DictionaryAttr::get(
          &context, {NamedAttribute("depth", I64Attr::get(&context, 12))}));

      auto provided = datapath.getFeature(require);
      REQUIRE(provided);
      CHECK_EQ(provided.getDepth(), 256);
    }

    SUBCASE("insufficient") {
      auto require = cast<feature::Queue>(DictionaryAttr::get(
          &context, {NamedAttribute("depth", I64Attr::get(&context, 512))}));

      auto provided = datapath.getFeature(require);
      REQUIRE_FALSE(provided);
    }
  }
}
