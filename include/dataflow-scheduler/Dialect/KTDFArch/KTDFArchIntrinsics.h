//===-- KTDFArchIntrinsics.h ------------------------------------*- c++ -*-===//
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
// This file defines the intrisics of the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_H_

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>

#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"

namespace mlir::ktdf_arch {

using GetAttrNameFn = StringAttr (KTDFArchDialect::*)() const;
using EmitErrorFn = function_ref<InFlightDiagnostic()>;

/// Base class for intrinsic attributes.
///
/// @tparam GetAttrName     Pointer to the cached name getter in the dialect.
/// @tparam AttrConstraint  Underlying attribute type constraint.
template <GetAttrNameFn GetAttrName, class AttrConstraint = Attribute>
struct IntrinsicAttr : AttrConstraint {
  static_assert(std::is_base_of_v<Attribute, AttrConstraint>);

  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    return isa<AttrConstraint>(attr);
  }

  using AttrConstraint::AttrConstraint;

  /// Gets the cached attribute name.
  [[nodiscard]] static auto getAttrName(MLIRContext* context) -> StringAttr {
    const auto* dialect = context->getLoadedDialect<KTDFArchDialect>();
    return (dialect->*GetAttrName)();
  }
  /// Gets the cached attribute name.
  [[nodiscard]] auto getAttrName() const -> StringAttr {
    return getAttrName(static_cast<const Attribute&>(*this).getContext());
  }
};

/// Base class for intrinsic features.
///
/// @tparam GetAttrName     Pointer to the cached name getter in the dialect.
template <GetAttrNameFn GetAttrName>
struct FeatureAttr : IntrinsicAttr<GetAttrName> {
  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    return isa<UnitAttr, DictionaryAttr>(attr);
  }

  using IntrinsicAttr<GetAttrName>::IntrinsicAttr;

  /// Determines whether this feature satisfies @p requirements .
  [[nodiscard]] auto test(Attribute requirements) const -> bool {
    return *this == requirements || isa<UnitAttr>(requirements);
  }

  /// Determines whether this feature has no nested attributes.
  [[nodiscard]] auto empty() const -> bool {
    return !*this || isa<UnitAttr>(*this) ||
           cast<DictionaryAttr>(*this).empty();
  }

  /// Obtains the nested @p Attr with @p name , if any.
  template <class Attr = Attribute>
  [[nodiscard]] auto getAttr(StringRef name) const -> Attr {
    const auto dict = dyn_cast_if_present<DictionaryAttr>(*this);
    return dict ? dict.template getAs<Attr>(name) : nullptr;
  }

  /// Obtains the value of the nested @p Attr with @p name , if any.
  template <class Attr = Attribute>
  [[nodiscard]] auto getValue(StringRef name) const
      -> std::optional<typename Attr::ValueType> {
    if (const auto attr = getAttr<Attr>(name); attr) {
      return attr.getValue();
    }

    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// Intrinsic Properties
//===----------------------------------------------------------------------===//

/// Indicates the bandwidth of a link in bytes per unit time.
struct BandwidthAttr
    : IntrinsicAttr<&KTDFArchDialect::getBandwidthAttrName, I64Attr> {
  using IntrinsicAttr::IntrinsicAttr;

  /// Gets the canonical BandwidthAttr for @p value .
  static auto get(MLIRContext* context, int64_t value) -> BandwidthAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;
};

/// Indicates the set of provided/required features of an operation.
struct FeaturesAttr
    : IntrinsicAttr<&KTDFArchDialect::getFeaturesAttrName, DictionaryAttr> {
  using IntrinsicAttr::IntrinsicAttr;

  auto verify(Operation* op) const -> LogicalResult;
};

/// Indicates the architecture element an operation is mapped to.
struct MapsToAttr
    : IntrinsicAttr<&KTDFArchDialect::getMapsToAttrName, Attribute> {
  using IntrinsicAttr::IntrinsicAttr;
};

/// Indicates a set of mutexes that allocation of this resource must obey.
struct OverlapsAttr
    : IntrinsicAttr<&KTDFArchDialect::getOverlapsAttrName, ArrayAttr> {
  using IntrinsicAttr::IntrinsicAttr;

  /// Determines whether @p attr is in the set.
  [[nodiscard]] auto contains(Attribute attr) const -> bool;

  /// Determines whether @p attrs overlap any in the set.
  [[nodiscard]] auto overlaps(ArrayRef<Attribute> attrs) const -> bool;
  /// Determines whether @p attr overlaps with the set.
  [[nodiscard]] auto overlaps(OverlapsAttr attr) const -> bool {
    return overlaps(attr.getValue());
  }
};

/// Indicates the permissible sizes of individual transfers on a link.
struct TransferGranularityAttr
    : IntrinsicAttr<&KTDFArchDialect::getTransferGranularityAttrName,
                    DenseI64ArrayAttr> {
  using IntrinsicAttr::IntrinsicAttr;

  /// Gets the canonical TransferGranularityAttr for @p value .
  static auto get(MLIRContext* context, ArrayRef<int64_t> value)
      -> TransferGranularityAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  /// Determines whether @p size is a permissible transfer size.
  [[nodiscard]] auto contains(int64_t size) const -> bool;
};

//===----------------------------------------------------------------------===//
// Intrinsic Features
//===----------------------------------------------------------------------===//

namespace feature {

/// Indicates that the execution unit can implement arbitrary programs.
struct Compute : FeatureAttr<&KTDFArchDialect::getFeatureComputeAttrName> {
  using FeatureAttr::FeatureAttr;
};

/// Indicates that the execution unit has SIMD lanes.
struct SIMD : FeatureAttr<&KTDFArchDialect::getFeatureSIMDAttrName> {
  using LanesAttr = TypedMapAttr<TypeAttr, I64Attr>;

  using FeatureAttr::FeatureAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  [[nodiscard]] auto test(SIMD requirements) const -> bool;

  /// Determines whether the unit can form splat vectors.
  [[nodiscard]] auto canSplat() const -> bool {
    return getAttr<UnitAttr>("splat") != nullptr;
  }

  /// Determines whether the unit can pad vectors with zeros.
  [[nodiscard]] auto canZeroPad() const -> bool {
    return getAttr<UnitAttr>("zero_pad") != nullptr;
  }

  /// Gets the number of lanes per scalar type.
  [[nodiscard]] auto getLanes() const -> LanesAttr {
    return getAttr<LanesAttr>("lanes");
  }
  /// Gets the number of lanes for @p scalar_type .
  ///
  /// @returns  Number of lanes for @p scalar_type , or 0 if not supported.
  [[nodiscard]] auto getLanes(Type scalar_type) const -> int64_t {
    if (const auto lanes = getLanes(); lanes) {
      return lanes.getValue(TypeAttr::get(scalar_type)).value_or(0);
    }
    return 0;
  }
};

/// Indicates that the link can have multiple transactions in flight.
struct Queue : FeatureAttr<&KTDFArchDialect::getFeatureQueueAttrName> {
  using FeatureAttr::FeatureAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  [[nodiscard]] auto test(Queue requirements) const -> bool;

  /// Gets a value indicating whether the transactions must complete in order.
  [[nodiscard]] auto isOrdered() const -> bool {
    return getAttr<UnitAttr>("ordered") != nullptr;
  }

  /// Gets a value indicating the total size of in flight transactions.
  [[nodiscard]] auto getSize() const -> std::optional<int64_t> {
    return getValue<I64Attr>("size");
  }

  /// Gets a value indicating how many transactions can be in flight.
  [[nodiscard]] auto getDepth() const -> std::optional<int64_t> {
    return getValue<I64Attr>("depth");
  }
};

}  // namespace feature

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_H_
