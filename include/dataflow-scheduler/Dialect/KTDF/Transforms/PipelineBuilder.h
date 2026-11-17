//===-- PipelineBuilder.h ---------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_

#include <llvm/ADT/PointerUnion.h>
#include <llvm/ADT/iterator_range.h>
#include <mlir/IR/Builders.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// RAII helper that allows moving code to a PipelineOp.
///
/// Users may call `insert` on ops to attempt moving them into the pipeline.
/// Some updating of the PipelineOp is deffered until the `build()` method is
/// called or the helper is destroyed.
///
/// Ops are moved to the start of their stage, and users are moved before their
/// producers, so the ops of each stage end up in program order. A value that
/// crosses from a producer stage P to a consumer stage C is sent through a pair
/// of a `ktdf.write_to_fifo` in P and a `ktdf.read_from_fifo` in C. All the
/// pairs from P to C share one physical FIFO, so only the order of the ops
/// links each value to its read. The builder places them as follows:
///
///  - The read goes at the top level of C, just before the earliest op that
///    contains a user of the value. Reads therefore follow their users, and
///    the reads for one op follow its operand order.
///  - The write goes among the trailing writes of P, before the first write to
///    C whose read comes after the new read. Writes therefore follow the reads.
///    Writes to other stages are not ordered against these, and are kept in
///    the order of their values where possible.
///  - The Allocator is told which slots of the P->C writes come after the new
///    write, so slots can follow the writes too.
///
/// None of this depends on the order in which ops are inserted or results are
/// sent. The IR this builds, and every pass that runs after it, must keep these
/// invariants:
///
///  I1. Each read runs under the same conditions, and the same number of times,
///      as its paired write. No pass may move an existing read or write into or
///      out of a conditional or loop. The builder only creates top-level pairs.
///  I2. In each P->C sequence, the writes and the reads are in the same order.
///  I3. In C, the reads are in the order of their users.
///  I4. The slots in an allocation are in the order of their writes and reads.
///  I5. Between two reads from P, there are only ops that don't wait on another
///      unit, so that nothing can deadlock against P's next write.
class PipelineBuilder : protected ImplicitLocOpBuilder {
 public:
  /// Controls the allocation of FIFO slots.
  struct Allocator {
    /// Gets the default allocator.
    [[nodiscard]] static auto getDefault() -> Allocator&;
    /// Gets the FifoSlotType between @p producer and @p consumer .
    [[nodiscard]] static auto getFifoSlotType(OpResult producer,
                                              StageOp consumer) -> FifoSlotType;

    /*implicit*/ Allocator() = default;
    /*implicit*/ Allocator(Allocator&&) = default;
    /*implicit*/ Allocator(const Allocator&) = default;
    auto operator=(Allocator&&) -> Allocator& = default;
    auto operator=(const Allocator&) -> Allocator& = default;

    virtual ~Allocator() = default;

    /// Determines whether a slot can be allocated for @p producer .
    [[nodiscard]] virtual auto canAllocate(OpResult producer) const -> bool;
    /// Allocates a slot for @p producer .
    ///
    /// @p later holds the slots of the writes from the producer's stage to
    /// @p consumer that come after the new write, in write order. An allocator
    /// that puts several slots in one allocation should place the new slot
    /// before the first of them it owns, or last if it owns none, so that its
    /// slots stay in write order.
    ///
    /// This method may not fail when `canAllocate` previously returned `true`
    /// for the given @p producer .
    [[nodiscard]] virtual auto allocate(PipelineBuilder& builder,
                                        OpResult producer, StageOp consumer,
                                        ArrayRef<FifoSlot> later) -> FifoSlot;
  };

  /// Determines where an operation should be placed in the pipeline.
  struct Placement {
    /// Computes the natural placement for @p op .
    ///
    /// If all users of @p op are in the same stage, this stage becomes the
    /// natural placement for @p op . Otherwise, the result is `nullptr`.
    [[nodiscard]] static auto natural(Operation* op) -> Placement;

    /// Initializes a not-into-pipleine placement.
    /*implicit*/ Placement() = default;
    /// @copydoc Placement()
    /*implicit*/ Placement(std::nullptr_t) : Placement() {}
    /// Initializes a Placement into @p stage .
    ///
    /// If @p erase_on_failure is set, the PipelineBuilder will erase the
    /// stage if the placement fails.
    ///
    /// @pre  `!erase_on_failure || stage.getBody()->empty()`
    /*implicit*/ Placement(StageOp stage, bool erase_on_failure = false)
        : stage(stage), erase_on_failure(erase_on_failure) {
      assert(!erase_on_failure || stage.getBody()->empty());
    }

    /// Gets whether the operation should be placed in the pipeline.
    explicit operator bool() const { return stage != nullptr; }
    /// Gets the stage the operation should be placed in.
    /*implicit*/ operator StageOp() const { return stage; }

    /// The stage the operation shall be inserted into.
    StageOp stage;
    /// Whether the stage shall be deleted if insertion fails.
    bool erase_on_failure = false;
  };

  /// Initializes a builder on a PipelineOp created using @p builder .
  explicit PipelineBuilder(OpBuilder& builder, Location loc,
                           Allocator* allocator = nullptr)
      : PipelineBuilder(PipelineOp::create(builder, loc), allocator,
                        builder.getListener()) {}
  explicit PipelineBuilder(ImplicitLocOpBuilder& builder,
                           Allocator* allocator = nullptr)
      : PipelineBuilder(PipelineOp::create(builder), allocator,
                        builder.getListener()) {}

  /// Ensures the PipelineOp is built.
  virtual ~PipelineBuilder() { build(); }

  PipelineBuilder(PipelineBuilder&&) = delete;
  PipelineBuilder(const PipelineBuilder&) = delete;
  auto operator=(PipelineBuilder&&) = delete;
  auto operator=(const PipelineBuilder&) = delete;

  //===--------------------------------------------------------------------===//
  // Builder Interface
  //===--------------------------------------------------------------------===//

  /// Gets the underlying MLIRContext.
  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return OpBuilder::getContext();
  }

  /// Converts @p attr to a units array.
  ///
  /// If @p attr is an ArrayAttr or `nullptr`, forwards it. Otherwise, wraps
  /// @p attr in an ArrayAttr and returns that.
  [[nodiscard]] auto getUnits(Attribute attr) const -> ArrayAttr;

  /// Gets the underlying PrivateBuilder.
  [[nodiscard]] auto getPrivateBuilder() -> PrivateBuilder& {
    return private_builder_;
  }
  /// Gets a builder for contained stages.
  [[nodiscard]] auto getStageBuilder() -> ImplicitLocOpBuilder { return *this; }

  /// Builds the PipelineOp.
  ///
  /// Applies all deferred modifications to the IR and finalizes the result.
  /// The PipelineBuilder is left in a state as if it was re-initialized on the
  /// resulting operation.
  auto build() -> PipelineOp;

  //===--------------------------------------------------------------------===//
  // Stage Building
  //===--------------------------------------------------------------------===//

  using stage_iterator = Block::op_iterator<StageOp>;
  using StageBuilderFn = function_ref<void(OpBuilder&, Location)>;

  /// Gets the stage for @p units , if it exists.
  [[nodiscard]] auto getStage(ArrayAttr units) const -> StageOp {
    return units_to_stage_.lookup(units);
  }
  /// Gets the stage for @p unit_or_units , if it exists.
  [[nodiscard]] auto getStage(Attribute unit_or_units) const -> StageOp {
    return getStage(getUnits(unit_or_units));
  }
  /// Gets the stage that @p op is in, if any.
  [[nodiscard]] auto getStage(Operation* op) const -> StageOp;

  /// Gets the stages in the pipeline.
  [[nodiscard]] auto getStages() const -> iterator_range<stage_iterator> {
    return getInsertionBlock()->getOps<StageOp>();
  }

  /// Creates a new stage.
  ///
  /// If @p units is not `nullptr`, the `applicable_units` will be set and the
  /// stage will be considered the new insertion point for that placement.
  virtual auto createStage(ArrayAttr units, std::optional<Location> loc,
                           StageBuilderFn body_builder) -> StageOp;
  /// Creates a new stage.
  ///
  /// See createStage(ArrayAttr, std::optional<Location>, StageBuilderFn) for
  /// more information.
  auto createStage(ArrayAttr units = nullptr,
                   std::optional<Location> loc = std::nullopt) -> StageOp {
    return createStage(units, loc, nullptr);
  }

  /// Gets or creates a stage for @p units .
  auto getOrCreateStage(ArrayAttr units,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    auto stage = getStage(units);
    return stage ? stage : createStage(units, loc);
  }
  /// Gets or creates a stage for @p unit_or_units .
  auto getOrCreateStage(Attribute unit_or_units,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    return getOrCreateStage(getUnits(unit_or_units), loc);
  }

  //===--------------------------------------------------------------------===//
  // FIFO Building
  //===--------------------------------------------------------------------===//

  /// Gets the underlying FIFO allocator.
  [[nodiscard]] auto getAllocator() const -> Allocator& { return *allocator_; }

  //===--------------------------------------------------------------------===//
  // Insertion & Placement
  //===--------------------------------------------------------------------===//

  using PlacementFn = function_ref<Placement(PipelineBuilder&, Operation*)>;

  /// Returns a Placement that will be `erased_on_failure`.
  auto tryPlacement(std::optional<Location> loc = std::nullopt) -> Placement {
    return {createStage({}, loc), true};
  }
  /// Returns a Placement for @p units that will be `erased_on_failure` if it
  /// was created.
  auto tryPlacement(ArrayAttr units, std::optional<Location> loc = std::nullopt)
      -> Placement {
    if (auto stage = getStage(units); stage) {
      return {stage, false};
    }
    return {createStage(units, loc), true};
  }
  /// Returns a Placement for @p unit_or_units that will be `erased_on_failure`
  /// if it was created.
  auto tryPlacement(Attribute unit_or_units,
                    std::optional<Location> loc = std::nullopt) -> Placement {
    return tryPlacement(getUnits(unit_or_units), loc);
  }

  /// Attempts to insert @p op into @p placement .
  virtual auto insert(Operation* op, Placement placement) -> LogicalResult;
  /// Attempts to insert @p ops into the pipeline.
  ///
  /// Runs a work list algorithm that attempts to put @p ops and all their
  /// transitive producers into the pipeline. Evalutes @p placement_fn for each
  /// eligible producer to determine the stage it should go to.
  ///
  /// Each step takes the latest pending op in program order, so every op is
  /// visited once, after all of its pending users. @p ops may be given in any
  /// order.
  void insert(ArrayRef<Operation*> ops, PlacementFn placement_fn);

 protected:
  explicit PipelineBuilder(PipelineOp pipeline, Allocator* allocator = nullptr,
                           OpBuilder::Listener* listener = nullptr);

  PrivateBuilder private_builder_;
  DenseMap<ArrayAttr, StageOp> units_to_stage_;
  DenseMap<StageOp, Token> tokens_;
  Allocator* allocator_;

 private:
  //===--------------------------------------------------------------------===//
  // FIFO Placement
  //===--------------------------------------------------------------------===//

  /// Gets an OpBuilder to move ops into @p stage .
  ///
  /// Ops go at the start of the stage, ahead of everything already there.
  [[nodiscard]] auto getOpBuilder(StageOp stage) -> OpBuilder;
  /// Gets an OpBuilder to insert a read of @p producer into @p consumer .
  ///
  /// The read goes at the top level of @p consumer , before the earliest op
  /// that contains a user of @p producer , or at the end if there is none.
  /// Reads that go before the same op follow the order of their first uses in
  /// it.
  [[nodiscard]] auto getReadBuilder(OpResult producer, StageOp consumer)
      -> OpBuilder;
  /// Gets an OpBuilder to insert a write of @p producer into its stage, paired
  /// with a read inserted by @p read_builder into @p consumer .
  ///
  /// The write goes among the trailing writes of the stage, after the writes
  /// to @p consumer whose read comes before the new read and before those
  /// whose read comes after it. Within those bounds, it goes before the first
  /// write of a value produced after @p producer , or as late as possible.
  /// @p later receives the slots of the writes to @p consumer that come after
  /// the new write, in order.
  ///
  /// @pre  `getStage(producer.getOwner())`
  [[nodiscard]] auto getWriteBuilder(OpResult producer, StageOp consumer,
                                     const OpBuilder& read_builder,
                                     SmallVectorImpl<FifoSlot>& later)
      -> OpBuilder;

  /// Adds a dependency on @p producer to @p consumer .
  ///
  /// @return Whether a new dependency was added.
  auto addDependency(StageOp producer, StageOp consumer) -> bool;

  /// Determines whether @p operand is available at its use.
  [[nodiscard]] auto isAvailable(const OpOperand& operand) const -> bool;
  /// Determines whether @p producer can be sent to another stage.
  [[nodiscard]] auto canSend(OpResult producer) const -> bool;
  /// Determines whether @p operand can be received from another stage.
  [[nodiscard]] auto canReceive(const OpOperand& operand) const -> bool;

  /// Sends @p producer to @p consumer via a FIFO and returns the read value.
  ///
  /// If @p producer is private, simply returns it (forwarding will be handled
  /// by the PrivateBuilder on `build()` or destruction). If @p consumer already
  /// reads @p producer ahead of all its users there, returns that read.
  /// Otherwise, a FIFO slot is allocated, @p producer is written to it, and the
  /// result is read from it (see getReadBuilder and getWriteBuilder).
  ///
  /// @pre  `getPrivateBuilder().isPrivate(producer.getOwner()) ||
  ///        canSend(producer)`
  auto send(OpResult producer, StageOp consumer) -> Value;
  /// Receives @p operand via a FIFO if needed and updates @p operand .
  ///
  /// If @p operand is already available, does nothing. Otherwise, the value is
  /// sent to the consumer stage and the operand is replaced with the read.
  ///
  /// @pre  `isAvailable(operand) || canReceive(operand)`
  void receive(OpOperand& operand);
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_
