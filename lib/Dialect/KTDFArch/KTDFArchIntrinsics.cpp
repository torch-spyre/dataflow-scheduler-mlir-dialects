//===-- KTDFArchIntrinsics.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>

#include <cstdint>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

auto emitIntrinsicError(Operation* op, const NamedAttribute& intrinsic)
    -> InFlightDiagnostic {
  return op->emitError("'") << intrinsic.getName().strref() << "' intrisic ";
}

}  // namespace

//===----------------------------------------------------------------------===//
// BandwidthAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyBandwidthAttr(Operation* op,
                                          const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (!isa<Link>(op)) {
    return emit_error() << "only valid on links";
  }

  const auto value = dyn_cast<BandwidthAttr>(attr.getValue());
  if (!value) {
    return emit_error() << "requires 64-bit integer attribute";
  }
  return value.verify(emit_error);
}

auto BandwidthAttr::get(MLIRContext* context, int64_t value) -> BandwidthAttr {
  return cast<BandwidthAttr>(I64Attr::get(context, value));
}

auto BandwidthAttr::verify(EmitErrorFn emit_error) const -> LogicalResult {
  if (getValue() <= 0) {
    return emit_error() << "value must be > 0";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FeaturesAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeaturesAttr(Operation* op,
                                         const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  const auto value = dyn_cast<FeaturesAttr>(attr.getValue());
  if (!value) {
    return emit_error() << "requires dictionary attribute";
  }
  return value.verify(op);
}

auto FeaturesAttr::verify(Operation* op) const -> LogicalResult {
  for (const auto& feature : getValue()) {
    const auto* iface =
        dyn_cast_if_present<FeatureDialectInterface>(feature.getNameDialect());
    if (!iface) {
      continue;
    }

    if (failed(iface->verify(op, feature))) {
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MapsToAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyMapsToAttr(Operation* op,
                                       const NamedAttribute& attr)
    -> LogicalResult {
  if (isa<KTDFArchDialect>(op->getDialect())) {
    return emitIntrinsicError(op, attr) << "only valid on mappable ops";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OverlapsAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyOverlapsAttr(Operation* op,
                                         const NamedAttribute& attr)
    -> LogicalResult {
  const auto value = dyn_cast<OverlapsAttr>(attr.getValue());
  if (!value) {
    return emitIntrinsicError(op, attr) << "requires array attribute";
  }
  return success();
}

auto OverlapsAttr::contains(Attribute attr) const -> bool {
  return llvm::is_contained(getValue(), attr);
}

auto OverlapsAttr::overlaps(ArrayRef<Attribute> attrs) const -> bool {
  return llvm::any_of(attrs,
                      [&](Attribute attr) -> bool { return contains(attr); });
}

//===----------------------------------------------------------------------===//
// TransferGranularityAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyTransferGranularityAttr(Operation* op,
                                                    const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (!isa<Link>(op)) {
    return emit_error() << "only valid on links";
  }

  const auto value = dyn_cast<TransferGranularityAttr>(attr.getValue());
  if (!value) {
    return emit_error() << "requires dense 64-bit integer array attribute";
  }
  return value.verify(emit_error);
}

auto TransferGranularityAttr::get(MLIRContext* context, ArrayRef<int64_t> value)
    -> TransferGranularityAttr {
  return cast<TransferGranularityAttr>(DenseI64ArrayAttr::get(context, value));
}

auto TransferGranularityAttr::verify(EmitErrorFn emit_error) const
    -> LogicalResult {
  llvm::SmallDenseSet<int64_t> values;
  for (auto value : asArrayRef()) {
    if (value < 0) {
      return emit_error() << "granularity must be >= 0";
    }
    if (!values.insert(value).second) {
      return emit_error() << "repeated granularity (" << value << ")";
    }
  }

  return success();
}

auto TransferGranularityAttr::contains(int64_t size) const -> bool {
  return llvm::is_contained(asArrayRef(), size);
}

//===----------------------------------------------------------------------===//
// feature::Compute
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureComputeAttr(Operation* op,
                                               const NamedAttribute& attr)
    -> LogicalResult {
  if (isa<KTDFArchDialect>(op->getDialect()) && !isa<ExecutionUnitOp>(op)) {
    return emitIntrinsicError(op, attr) << "only valid on execution units";
  }

  return success();
}

auto KTDFArchDialect::testFeatureCompute(Attribute provided,
                                         const Feature& required) -> bool {
  return provided == required.getValue();
}

//===----------------------------------------------------------------------===//
// feature::SIMD
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureSIMDAttr(Operation* op,
                                            const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (isa<KTDFArchDialect>(op->getDialect()) && !isa<ExecutionUnitOp>(op)) {
    return emit_error() << "only valid on execution units";
  }

  const auto value = dyn_cast<feature::SIMD>(attr.getValue());
  if (!value) {
    return emit_error() << "requires unit or dictionary attribute";
  }
  return value.verify(emit_error);
}

auto feature::SIMD::verify(EmitErrorFn emit_error) const -> LogicalResult {
  const auto splat = getAttr("splat");
  if (splat && !isa<UnitAttr>(splat)) {
    return emit_error() << "'splat' requires unit attribute";
  }

  const auto zero_pad = getAttr("zero_pad");
  if (zero_pad && !isa<UnitAttr>(zero_pad)) {
    return emit_error() << "'zero_pad' requires unit attribute";
  }

  const auto lanes = getAttr("lanes");
  if (lanes && !isa<LanesAttr>(lanes)) {
    return emit_error() << "'lanes' requires '" << MapAttr::getMnemonic()
                        << "' from type to 64-bit integer attributes";
  }

  return success();
}

auto KTDFArchDialect::testFeatureSIMD(Attribute provided,
                                      const Feature& required) -> bool {
  const auto required_value = cast<feature::SIMD>(required.getValue());
  const auto provided_value = cast<feature::SIMD>(provided);

  return provided_value.test(required_value);
}

auto feature::SIMD::test(feature::SIMD requirements) const -> bool {
  if (requirements.canSplat() && !canSplat()) {
    return false;
  }

  if (requirements.canZeroPad() && !canZeroPad()) {
    return false;
  }

  if (const auto required = requirements.getLanes(); required) {
    const auto required_lanes = required.getEntries();
    const auto provided_lanes = getLanes();

    if (!provided_lanes && !required_lanes.empty()) {
      return false;
    }

    if (llvm::any_of(required_lanes, [&](const auto& require) -> bool {
          return provided_lanes.getValue(require.first) <
                 require.second.getValue();
        })) {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// feature::Queue
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureQueueAttr(Operation* op,
                                             const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (isa<KTDFArchDialect>(op->getDialect()) && !isa<Link>(op)) {
    return emit_error() << "only valid on links";
  }

  const auto value = dyn_cast<feature::Queue>(attr.getValue());
  if (!value) {
    return emit_error() << "requires unit or dictionary attribute";
  }
  return value.verify(emit_error);
}

auto feature::Queue::verify(EmitErrorFn emit_error) const -> LogicalResult {
  const auto ordered = getAttr("ordered");
  if (ordered && !isa<UnitAttr>(ordered)) {
    return emit_error() << "'ordered' requires unit attribute";
  }

  if (const auto maybe_size = getSize(); maybe_size) {
    if (*maybe_size <= 0) {
      return emit_error() << "'size' must be > 0";
    }
  } else if (getAttr("size")) {
    return emit_error() << "'size' requires 64-bit integer attribute";
  }

  if (const auto maybe_depth = getDepth(); maybe_depth) {
    if (*maybe_depth <= 0) {
      return emit_error() << "'depth' must be > 0";
    }
  } else if (getAttr("depth")) {
    return emit_error() << "'depth' requires 64-bit integer attribute";
  }

  return success();
}

auto KTDFArchDialect::testFeatureQueue(Attribute provided,
                                       const Feature& required) -> bool {
  const auto required_value = cast<feature::Queue>(required.getValue());
  const auto provided_value = cast<feature::Queue>(provided);

  return provided_value.test(required_value);
}

auto feature::Queue::test(Queue requirements) const -> bool {
  if (requirements.isOrdered() && !isOrdered()) {
    return false;
  }

  if (const auto required = requirements.getSize(); required) {
    const auto provided = getSize();
    if (!provided || provided < required) {
      return false;
    }
  }

  if (const auto required = requirements.getDepth(); required) {
    const auto provided = getDepth();
    if (!provided || provided < required) {
      return false;
    }
  }

  return true;
}
