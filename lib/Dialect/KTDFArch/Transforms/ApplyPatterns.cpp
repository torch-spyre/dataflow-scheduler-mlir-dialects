//===-- ApplyPatterns.cpp ---------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
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

#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/ApplyPatterns.h"

#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDL/IR/PDLOps.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h" // IWYU pragma: keep

#define DEBUG_TYPE "ktdfarch-apply-patterns"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace mlir::ktdf_arch {
#define GEN_PASS_DEF_APPLYPATTERNSPASS
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h.inc"
} // namespace mlir::ktdf_arch

//===----------------------------------------------------------------------===//
// Native Constraints and Rewrites
//===----------------------------------------------------------------------===//

namespace {

// Returns the integer ordinal of a spyreop.reduction_scope attribute, or
// std::nullopt if the attribute is not of that type.
// The ordinal values match the TableGen definition: InSlice=0, AcrossSlice=1.
// Using standard MLIR APIs avoids a hard dependency on KTIR headers.
auto getReductionScopeOrdinal(Attribute attr) -> std::optional<int64_t> {
  if (!attr)
    return std::nullopt;
  if (attr.getAbstractAttribute().getName() != "spyreop.reduction_scope")
    return std::nullopt;
  return cast<IntegerAttr>(attr).getInt();
}

// Native constraint: succeeds iff the attribute is reduction_scope<in_slice>.
auto spyreIsInSlice(PatternRewriter & /*rewriter*/, PDLResultList & /*results*/,
                    ArrayRef<PDLValue> values) -> LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<Attribute>());
  return success(ordinal && *ordinal == 0); // InSlice = 0
}

// Native constraint: succeeds iff the attribute is
// reduction_scope<across_slice>.
auto spyreIsAcrossSlice(PatternRewriter & /*rewriter*/,
                        PDLResultList & /*results*/, ArrayRef<PDLValue> values)
    -> LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<Attribute>());
  return success(ordinal && *ordinal == 1); // AcrossSlice = 1
}

auto ktdfArchMappedTo(PatternRewriter & /*rewriter*/, PDLResultList &results,
                      ArrayRef<PDLValue> values) -> LogicalResult {
  assert(values.size() == 1);

  auto *const op = values[0].cast<Operation *>();
  const auto maps_to = getProperty<MapsToAttr>(op);
  if (!maps_to) {
    return failure();
  }

  results.push_back(maps_to);
  return success();
}

auto ktdfArchHasFeature(PatternRewriter & /*rewriter*/, PDLResultList &results,
                        ArrayRef<PDLValue> values) -> LogicalResult {
  assert(values.size() >= 2 && values.size() <= 3);

  auto *const op = values[0].cast<Operation *>();
  const auto name = cast<StringAttr>(values[1].cast<Attribute>());

  std::optional<Feature> provided;
  if (values.size() == 2) {
    provided = getFeature(op, name);
  } else {
    provided = getFeature(op, {name, values[2].cast<Attribute>()});
  }

  if (!provided) {
    return failure();
  }

  results.push_back(provided->getValue());
  return success();
}

void ktdfArchMapTo(PatternRewriter &rewriter, Operation *op,
                   Attribute maps_to) {
  rewriter.modifyOpInPlace(
      op, [&]() { setProperty(op, cast<MapsToAttr>(maps_to)); });
}

void ktdfArchSetFeature(PatternRewriter &rewriter, Operation *op,
                        Attribute name, Attribute value) {
  rewriter.modifyOpInPlace(op, [&]() {
    NamedAttrList features(
        static_cast<DictionaryAttr>(getProperty<FeaturesAttr>(op)));
    features.set(cast<StringAttr>(name), value);
    setProperty(op,
                cast<FeaturesAttr>(features.getDictionary(op->getContext())));
  });
}

// Native rewrite: create an agen.vector_load using the C++ builder so that
// AttrSizedOperandSegments is set correctly.
// Signature (PDL values): mem_ref, map_op_0..N, affine_map, load_set,
// load_order, result_type values[0]=mem_ref, values[1..N-4]=map_operands,
// values[N-3]=map_attr, values[N-2]=load_set_attr, values[N-1]=load_order_attr,
// values[N]=result_type
auto agenVectorLoad(PatternRewriter &rewriter, PDLResultList &results,
                    ArrayRef<PDLValue> values) -> LogicalResult {
  if (values.size() < 4)
    return failure();

  auto mem_ref_val = values[0].cast<Value>();
  const size_t num_attrs = 4; // map, set, order, type
  const size_t num_map_ops = values.size() - 1 - num_attrs;

  SmallVector<Value> map_operands;
  for (size_t i = 1; i <= num_map_ops; ++i)
    map_operands.push_back(values[i].cast<Value>());

  const size_t base = 1 + num_map_ops;
  auto map_attr = cast<AffineMapAttr>(values[base + 0].cast<Attribute>());
  auto set_attr = cast<IntegerSetAttr>(values[base + 1].cast<Attribute>());
  auto order_attr = cast<AffineMapAttr>(values[base + 2].cast<Attribute>());
  auto result_ty = cast<VectorType>(values[base + 3].cast<Type>());

  auto op = agen::VectorLoadOp::create(
      rewriter, rewriter.getUnknownLoc(), result_ty, mem_ref_val,
      /*dbgName=*/StringAttr{}, map_attr.getAffineMap(), map_operands, set_attr,
      order_attr.getAffineMap(), /*multicast_info=*/Value{});

  results.push_back(op.getResult());
  return success();
}

// Native rewrite: create a vectorchain.shuffle using the C++ builder so that
// AttrSizedOperandSegments is set correctly (variable=[], pad=[], mask=none).
// Signature (PDL values): input, indices_attr, repetition_attr, result_type
auto vectorchainShuffle(PatternRewriter &rewriter, PDLResultList &results,
                        ArrayRef<PDLValue> values) -> LogicalResult {
  if (values.size() != 4)
    return failure();

  auto input = values[0].cast<Value>();
  auto indices = cast<ArrayAttr>(values[1].cast<Attribute>());
  auto repetition = cast<IntegerAttr>(values[2].cast<Attribute>());
  auto result_ty = cast<VectorType>(values[3].cast<Type>());

  auto op = vectorchain::ShuffleOp::create(
      rewriter, rewriter.getUnknownLoc(), result_ty, input, indices, repetition,
      /*dbgName=*/StringAttr{});

  results.push_back(op.getResult());
  return success();
}

} // namespace

void ktdf_arch::registerNativeFunctions(PDLPatternModule &patterns) {
  // Constraint functions
  patterns.registerConstraintFunction("ktdf_arch.mapped_to", ktdfArchMappedTo);
  patterns.registerConstraintFunction("ktdf_arch.has_feature",
                                      ktdfArchHasFeature);

  // Rewrite functions
  patterns.registerRewriteFunction("ktdf_arch.map_to", ktdfArchMapTo);
  patterns.registerRewriteFunction("ktdf_arch.set_feature", ktdfArchSetFeature);
  patterns.registerRewriteFunction("agen.vector_load", agenVectorLoad);
  patterns.registerRewriteFunction("vectorchain.shuffle", vectorchainShuffle);

  // Scope constraints for spyreop.slice_reduction dispatch
  patterns.registerConstraintFunction("spyreop.is_in_slice", spyreIsInSlice);
  patterns.registerConstraintFunction("spyreop.is_across_slice",
                                      spyreIsAcrossSlice);
}

//===----------------------------------------------------------------------===//
// PatternGroups
//===----------------------------------------------------------------------===//

auto PatternGroups::contains(StringRef group) const -> bool {
  const auto *const it = llvm::lower_bound(groups_, group);
  return it != groups_.end() && *it == group;
}

void PatternGroups::initialize(SmallVectorImpl<std::string> &groups) {
  llvm::raw_svector_ostream os(key_);
  llvm::sort(groups);
  llvm::interleaveComma(groups, os, [&](StringRef group) {
    auto *const begin = key_.end();
    key_.append(group);
    groups_.emplace_back(begin, group.size());
  });
}

//===----------------------------------------------------------------------===//
// ktdf_arch::getPatterns
//===----------------------------------------------------------------------===//

namespace {

auto clonePatterns(PatternsOp from, PDLPatternModule &to) -> size_t {
  size_t result = 0U;

  OpBuilder builder(to.getModule().getBodyRegion());
  for (auto pattern : from.getOps<pdl::PatternOp>()) {
    builder.clone(*pattern);
    ++result;
  }

  return result;
}

} // namespace

auto ktdf_arch::getPatterns(const Device &device, PDLPatternModule &patterns,
                            const PatternGroups &enabled_groups) -> size_t {
  // Ensure there is a module to clone into.
  if (!patterns.getModule()) {
    patterns.mergeIn(PDLPatternModule(ModuleOp::create(device.getLoc())));
  }

  // Clone the selected patterns into the module. Since PDLPatternModule needs
  // ownership of the pattern ops, there is no better way of caching these.
  size_t result = 0U;
  for (auto pattern_set : device.getBodyRegion().getOps<PatternsOp>()) {
    const auto groups = pattern_set.getGroups().getAsValueRange<StringAttr>();
    if (!groups.empty() && llvm::none_of(groups, [&](StringRef group) -> bool {
          return enabled_groups.contains(group);
        })) {
      continue;
    }

    result += clonePatterns(pattern_set, patterns);
  }

  registerNativeFunctions(patterns);
  return result;
}

//===----------------------------------------------------------------------===//
// PatternCache
//===----------------------------------------------------------------------===//

auto PatternCache::get(const PatternGroups &enabled_groups)
    -> FrozenRewritePatternSet {
  llvm::sys::SmartScopedLock<true> lock(mutex_);

  if (const auto it = map_.find(enabled_groups); it != map_.end()) {
    return it->second;
  }

  PDLPatternModule pdl_patterns;
  const auto num_patterns =
      getPatterns(getDevice(), pdl_patterns, enabled_groups);

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "selecting " << num_patterns << " pattern(s)";
    if (!enabled_groups.empty()) {
      os << " in group(s) ";
      llvm::interleaveComma(enabled_groups, os);
    }
    os << " of device '" << getDevice().getName() << "'";
  });

  FrozenRewritePatternSet result;
  if (num_patterns > 0) {
    result = FrozenRewritePatternSet(std::move(pdl_patterns));
  }

  return map_[enabled_groups] = result;
}

//===----------------------------------------------------------------------===//
// ApplyPatternsPass
//===----------------------------------------------------------------------===//

namespace {

struct ApplyPatternsPass
    : public ktdf_arch::impl::ApplyPatternsPassBase<ApplyPatternsPass> {
  using ApplyPatternsPassBase::ApplyPatternsPassBase;

  void runOnOperation() override {
    // Find the device that this function maps to.
    auto declaration = findDeviceDeclarationFor(getOperation());
    if (!declaration) {
      return;
    }
    DeviceRef device(declaration, getAnalysisManager());
    LDBG() << "processing "
           << OpWithFlags(getOperation(), OpPrintingFlags().skipRegions())
           << " with device " << declaration.getName();

    // Get the (cached) rewrite pattern set. This prevents cloning the PDL
    // module
    const auto patterns = device.getOrCreateView<PatternCache>().get(
        PatternGroups(llvm::from_range, enabled_groups));

    // Run all the patterns.
    auto changed = false;
    if (failed(applyPatternsGreedily(getOperation(), patterns,
                                     GreedyRewriteConfig(), &changed))) {
      signalPassFailure();
      return;
    }

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

} // namespace

auto ktdf_arch::createApplyPatternsPass(
    std::initializer_list<StringRef> enabled_groups) -> std::unique_ptr<Pass> {
  ApplyPatternsPassOptions options;
  for (auto group : enabled_groups) {
    options.enabled_groups.emplace_back(group.str());
  }
  return createApplyPatternsPass(options);
}
