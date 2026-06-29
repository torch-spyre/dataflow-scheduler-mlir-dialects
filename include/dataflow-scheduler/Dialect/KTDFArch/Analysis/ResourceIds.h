//===-- ResourceIds.h -------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_RESOURCEIDS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_RESOURCEIDS_H_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/AnalysisManager.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

namespace mlir::ktdf_arch {

/// Analysis that manages the device-unique identifiers of resources.
class ResourceIds : public DeviceView {
  using map_type = DenseMap<StringAttr, Resource>;

 public:
  /// Creates a ResourceLookup for the device declared by @p declaration .
  explicit ResourceIds(DeviceOp declaration, AnalysisManager& analyses);

  /// Obtains the resource with @p id , if it exists.
  [[nodiscard]] auto lookup(StringAttr id) -> Resource {
    return map_.lookup(id);
  }
  /// @copydoc lookup(StringAttr)
  [[nodiscard]] auto lookup(StringRef id) -> Resource {
    return lookup(StringAttr::get(getContext(), id));
  }

  /// Assigns @p id to @p resource if possible.
  ///
  /// @retval false @p resource is outside the device or @p id is already taken.
  /// @retval true  @p resource was assigned @p id .
  auto assign(Resource resource, StringAttr id) -> bool;
  /// Assigns a unique identifier starting with @p prefix to @p resource .
  ///
  /// @retval nullptr     @p resource is outside the device.
  /// @retval StringAttr  Identifier that was assigned to @p resource .
  auto assign(Resource resource, StringRef prefix) -> StringAttr;

  /// Gets the unique identifier of @p resource , assigning one if necessary.
  ///
  /// @retval nullptr     @p resource is outside the device.
  /// @retval StringAttr  Unique identifier of @p resource .
  auto getOrAssign(Resource resource,
                   std::optional<StringRef> prefix = std::nullopt)
      -> StringAttr;

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = map_type::value_type;
  using size_type = map_type::size_type;
  using iterator = map_type::const_iterator;

  [[nodiscard]] auto empty() const -> bool { return map_.empty(); }
  [[nodiscard]] auto size() const -> size_type { return map_.size(); }

  [[nodiscard]] auto begin() const -> iterator { return map_.begin(); }
  [[nodiscard]] auto end() const -> iterator { return map_.end(); }

 private:
  map_type map_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_RESOURCEIDS_H_
