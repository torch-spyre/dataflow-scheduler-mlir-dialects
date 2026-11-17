//===-- PipelineBuilder.cpp -------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/BreadthFirstIterator.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <algorithm>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"

#define DEBUG_TYPE "ktdf-pipeline-builder"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

const auto kSkipRegions = OpPrintingFlags().skipRegions();

/// Determines whether @p lhs comes before @p rhs in program order.
///
/// Compares the ancestors of @p lhs and @p rhs that are siblings under their
/// closest common ancestor. An op comes before the ops nested in it.
///
/// @pre  @p lhs and @p rhs are nested in the same op.
[[nodiscard]] auto isBeforeInProgram(Operation* lhs, Operation* rhs) -> bool {
  SmallVector<Operation*> lhs_path;
  SmallVector<Operation*> rhs_path;
  for (auto* op = lhs; op != nullptr; op = op->getParentOp()) {
    lhs_path.push_back(op);
  }
  for (auto* op = rhs; op != nullptr; op = op->getParentOp()) {
    rhs_path.push_back(op);
  }

  // Drop the common ancestors, outermost first.
  while (!lhs_path.empty() && !rhs_path.empty() &&
         lhs_path.back() == rhs_path.back()) {
    lhs_path.pop_back();
    rhs_path.pop_back();
  }
  if (lhs_path.empty() || rhs_path.empty()) {
    // One contains the other (or they are the same op).
    return !rhs_path.empty();
  }

  auto* const lhs_sibling = lhs_path.back();
  auto* const rhs_sibling = rhs_path.back();
  auto* const lhs_block = lhs_sibling->getBlock();
  auto* const rhs_block = rhs_sibling->getBlock();
  assert(lhs_block && rhs_block && "ops must be nested in the same op");
  if (lhs_block == rhs_block) {
    return lhs_sibling->isBeforeInBlock(rhs_sibling);
  }
  if (lhs_block->getParent() != rhs_block->getParent()) {
    return lhs_block->getParent()->getRegionNumber() <
           rhs_block->getParent()->getRegionNumber();
  }
  for (auto& block : *lhs_block->getParent()) {
    if (&block == lhs_block || &block == rhs_block) {
      return &block == lhs_block;
    }
  }
  llvm_unreachable("block not in its region");
}

/// Determines whether @p lhs comes before @p rhs in program order, ordering
/// uses in the same op by operand number.
[[nodiscard]] auto isBeforeInProgram(OpOperand& lhs, OpOperand& rhs) -> bool {
  if (lhs.getOwner() == rhs.getOwner()) {
    return lhs.getOperandNumber() < rhs.getOperandNumber();
  }
  return isBeforeInProgram(lhs.getOwner(), rhs.getOwner());
}

/// Gets the earliest top-level op in @p block that is or contains a user of
/// @p value , if any.
[[nodiscard]] auto getFirstUserIn(Value value, Block& block) -> Operation* {
  Operation* result = nullptr;
  for (auto* const user : value.getUsers()) {
    if (auto* const op = block.findAncestorOpInBlock(*user);
        op && (!result || op->isBeforeInBlock(result))) {
      result = op;
    }
  }
  return result;
}

/// Gets the first use of @p value in @p op or the ops nested in it, if any.
[[nodiscard]] auto getFirstUseIn(Value value, Operation* op) -> OpOperand* {
  OpOperand* result = nullptr;
  for (auto& use : value.getUses()) {
    if (op->isAncestor(use.getOwner()) &&
        (!result || isBeforeInProgram(use, *result))) {
      result = &use;
    }
  }
  return result;
}

}  // namespace

//===----------------------------------------------------------------------===//
// PipelineBuilder::Allocator
//===----------------------------------------------------------------------===//

namespace {

[[nodiscard]] auto getUnitOrUnits(StageOp stage) -> Attribute {
  const auto units = stage.getApplicableUnitsAttr();
  if (!units) {
    return ArrayAttr::get(stage->getContext(), {});
  }

  return units.size() == 1 ? units.getValue().front() : units;
}

}  // namespace

auto PipelineBuilder::Allocator::getDefault() -> Allocator& {
  static Allocator instance;
  return instance;
}

auto PipelineBuilder::Allocator::getFifoSlotType(OpResult producer,
                                                 StageOp consumer)
    -> FifoSlotType {
  const auto value_type = cast<ShapedType>(producer.getType());
  const auto producer_unit =
      getUnitOrUnits(producer.getOwner()->getParentOfType<StageOp>());
  const auto consumer_unit = getUnitOrUnits(consumer);

  // Create the appropriate type for the slot, which flattens the elements.
  return FifoSlotType::get(producer.getContext(), producer_unit, consumer_unit,
                           value_type.getNumElements(),
                           value_type.getElementType());
}

auto PipelineBuilder::Allocator::canAllocate(OpResult producer) const -> bool {
  const auto value_type = dyn_cast<ShapedType>(producer.getType());
  return value_type && !isa<MemRefType>(value_type) &&
         value_type.hasStaticShape();
}

// Each slot is its own allocation, so there is no slot order to keep.
auto PipelineBuilder::Allocator::allocate(PipelineBuilder& builder,
                                          OpResult producer, StageOp consumer,
                                          ArrayRef<FifoSlot> /*later*/)
    -> TypedValue<FifoSlotType> {
  return cast<TypedValue<FifoSlotType>>(
      builder.getPrivateBuilder()
          .createFifo({getFifoSlotType(producer, consumer)}, {},
                      consumer->getLoc())
          .front());
}

//===----------------------------------------------------------------------===//
// PipelineBuilder::Placement
//===----------------------------------------------------------------------===//

auto PipelineBuilder::Placement::natural(Operation* op) -> Placement {
  StageOp result;
  for (auto* const user : op->getUsers()) {
    auto consumer_stage = user->getParentOfType<StageOp>();
    if (!consumer_stage || (result && consumer_stage != result)) {
      return nullptr;
    }

    result = consumer_stage;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// PipelineBuilder
//===----------------------------------------------------------------------===//

auto PipelineBuilder::getUnits(Attribute attr) const -> ArrayAttr {
  auto units = dyn_cast_if_present<ArrayAttr>(attr);
  if (!units && attr) {
    units = ArrayAttr::get(getContext(), {attr});
  }

  return units;
}

auto PipelineBuilder::getOpBuilder(StageOp stage) -> OpBuilder {
  auto& body = *stage.getBody();
  return OpBuilder(&body, body.begin(), listener);
}

// Finds, among the users of the producer in the consumer, the one whose
// top-level ancestor comes first, and inserts before that ancestor. A user
// nested in an `scf.if` or `scf.for` thus gets its read before that op, so the
// read is unconditional like its write. Since ops are only ever added at the
// start of a stage, nothing inserted later can land between a read and its
// users.
//
// Several values may be read for the same ancestor, e.g. the operands of one
// op. Those reads sit right before the ancestor, and the new read goes before
// the ones whose first use in the ancestor comes after its own. Reads for one
// op thus follow its operand order, whatever order they are sent in.
auto PipelineBuilder::getReadBuilder(OpResult producer, StageOp consumer)
    -> OpBuilder {
  auto& body = *consumer.getBody();
  auto* const first_user = getFirstUserIn(producer, body);
  if (!first_user) {
    return OpBuilder(&body, body.end(), listener);
  }

  auto& first_use = *getFirstUseIn(producer, first_user);
  auto* read_before = first_user;
  for (auto read =
           dyn_cast_if_present<ReadFromFifoOp>(first_user->getPrevNode());
       read; read = dyn_cast_if_present<ReadFromFifoOp>(read->getPrevNode())) {
    auto* const use = getFirstUseIn(read, first_user);
    if (!use || !isBeforeInProgram(first_use, *use)) {
      break;
    }
    read_before = read;
  }

  return OpBuilder(read_before, listener);
}

namespace {

/// Finds the top-level `ktdf.read_from_fifo` of @p slot in @p block , if any.
[[nodiscard]] auto findReadOf(FifoSlot slot, Block& block) -> ReadFromFifoOp {
  for (auto* const user : slot.getUsers()) {
    if (auto read = dyn_cast<ReadFromFifoOp>(user);
        read && read->getBlock() == &block) {
      return read;
    }
  }

  return nullptr;
}

/// Finds a top-level `ktdf.read_from_fifo` of @p value in @p block that comes
/// before @p before , or anywhere in @p block if @p before is `nullptr`.
[[nodiscard]] auto findReadIn(Value value, Block& block, Operation* before)
    -> ReadFromFifoOp {
  for (auto* const user : value.getUsers()) {
    auto write = dyn_cast<WriteToFifoOp>(user);
    if (!write || value != write.getData()) {
      continue;
    }
    if (auto read = findReadOf(write.getFifoSlot(), block);
        read && (!before || read->isBeforeInBlock(before))) {
      return read;
    }
  }

  return nullptr;
}

}  // namespace

// Walks the top-level writes of the producer stage. Those whose read is in the
// consumer form the P->C sequence, which is kept sorted by the position of the
// reads: the new write must go after the ones whose read comes before the new
// read, and before the ones whose read comes at or after it. Writes to other
// stages are not ordered against this sequence, so within those bounds the new
// write goes before the first write whose value is produced after its own.
// Writes thus follow program order where the P->C sequence allows it.
auto PipelineBuilder::getWriteBuilder(OpResult producer, StageOp consumer,
                                      const OpBuilder& read_builder,
                                      SmallVectorImpl<FifoSlot>& later)
    -> OpBuilder {
  auto producer_stage = getStage(producer.getOwner());
  assert(producer_stage);
  auto& body = *producer_stage.getBody();
  assert(read_builder.getInsertionBlock() == consumer.getBody());
  const auto read_point = read_builder.getInsertionPoint();
  auto* const new_read =
      read_point == consumer.getBody()->end() ? nullptr : &*read_point;
  auto* const new_value = body.findAncestorOpInBlock(*producer.getOwner());

  // Once a write of the sequence comes after the new one, `write_before` is set
  // and, since the sequence is sorted, all its later writes come after too.
  Operation* write_before = nullptr;
  for (auto write : body.getOps<WriteToFifoOp>()) {
    auto read = findReadOf(write.getFifoSlot(), *consumer.getBody());
    if (read && new_read && !read->isBeforeInBlock(new_read)) {
      // This write comes after the new one.
      if (!write_before) {
        write_before = write;
      }
      later.push_back(write.getFifoSlot());
    } else if (read) {
      // The new write must come after this one.
      write_before = nullptr;
    } else if (!write_before && new_value) {
      // A write to another stage: keep the order of the values.
      auto* const value = write.getData().getDefiningOp();
      auto* const value_op =
          value ? body.findAncestorOpInBlock(*value) : nullptr;
      if (value_op && new_value->isBeforeInBlock(value_op)) {
        write_before = write;
      }
    }
  }

  return write_before ? OpBuilder(write_before, listener)
                      : OpBuilder(&body, body.end(), listener);
}

auto PipelineBuilder::build() -> PipelineOp {
  // Build the PrivateOp and erase it if it's empty.
  if (auto private_op = private_builder_.build();
      private_op && private_op.getBody()->without_terminator().empty()) {
    private_builder_.erase();
  }

  return cast<PipelineOp>(getInsertionBlock()->getParentOp());
}

namespace {

/// Gets the stage @p op is in inside @p pipeline , if any.
[[nodiscard]] auto getStage(Block* pipeline, Operation* op) -> StageOp {
  for (auto stage = op->getParentOfType<StageOp>(); stage;
       stage = stage->getParentOfType<StageOp>()) {
    if (stage->getBlock() == pipeline) {
      return stage;
    }
  }

  return nullptr;
}

}  // namespace

auto PipelineBuilder::getStage(Operation* op) const -> StageOp {
  return ::getStage(getInsertionBlock(), op);
}

auto PipelineBuilder::createStage(ArrayAttr units, std::optional<Location> loc,
                                  StageBuilderFn body_builder) -> StageOp {
  // Insert stages after the PrivateOp, so that producer fusion neatly orders
  // the stages in topological order.
  OpBuilder::InsertionGuard guard(*this);
  setInsertionPointAfter(private_builder_.getInsertionPoint()->getParentOp());

  auto result =
      StageOp::create(*this, loc.value_or(getLoc()), {}, {}, body_builder);
  if (units) {
    result.setApplicableUnitsAttr(units);
    units_to_stage_[units] = result;
  }
  return result;
}

auto PipelineBuilder::addDependency(StageOp producer, StageOp consumer)
    -> bool {
  if (producer->getBlock() != getInsertionBlock() ||
      consumer->getBlock() != getInsertionBlock()) {
    return false;
  }

  IRRewriter rewriter(*this);
  auto token = tokens_.lookup(producer);
  if (!token) {
    tokens_[producer] = token =
        private_builder_.createToken(producer->getLoc());
    rewriter.modifyOpInPlace(producer,
                             [&]() { producer.addOutDependency(token); });
  }

  rewriter.startOpModification(consumer);
  if (!consumer.addInDependency(token)) {
    rewriter.cancelOpModification(consumer);
    return false;
  }
  rewriter.finalizeOpModification(consumer);
  return true;
}

auto PipelineBuilder::isAvailable(const OpOperand& operand) const -> bool {
  auto* const definition = operand.get().getParentRegion();
  for (auto* region = operand.getOwner()->getParentRegion(); region != nullptr;
       region = region->getParentRegion()) {
    if (region == definition) {
      return true;
    }
    if (region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return false;
    }
  }

  auto producer = dyn_cast<OpResult>(operand.get());
  return producer && private_builder_.isPrivate(producer.getOwner());
}

auto PipelineBuilder::canSend(OpResult producer) const -> bool {
  auto stage = getStage(producer.getOwner());
  return stage && allocator_->canAllocate(producer);
}

auto PipelineBuilder::canReceive(const OpOperand& operand) const -> bool {
  auto stage = getStage(operand.getOwner());
  auto producer = dyn_cast<OpResult>(operand.get());
  return stage && producer && canSend(producer);
}

// Reuses a read of the producer in the consumer if it comes before all the
// remaining users there. Otherwise, places a new read before those users, a new
// write in the producer stage in the same order as the reads, and asks the
// allocator for a slot in the same order as the writes.
auto PipelineBuilder::send(OpResult producer, StageOp consumer) -> Value {
  if (private_builder_.isPrivate(producer.getOwner())) {
    // The producer is a private op, which all stages have access to.
    return producer;
  }

  assert(canSend(producer));
  auto producer_stage = getStage(producer.getOwner());
  assert(producer_stage);

  // Lookup existing FIFO reads that produce this value ahead of its users.
  auto read = findReadIn(producer, *consumer.getBody(),
                         getFirstUserIn(producer, *consumer.getBody()));
  if (!read) {
    auto read_builder = getReadBuilder(producer, consumer);
    SmallVector<FifoSlot> later;
    auto write_builder =
        getWriteBuilder(producer, consumer, read_builder, later);

    // Allocate a new FIFO slot for this value.
    const auto slot = allocator_->allocate(*this, producer, consumer, later);
    assert(slot && "allocator may not fail");

    WriteToFifoOp::create(write_builder, consumer->getLoc(), producer, slot);

    // On the consumer side, create a new read from the slot.
    read = ReadFromFifoOp::create(read_builder, consumer->getLoc(),
                                  producer.getType(), slot);

    // Introduce a dependency between producer and consumer via private tokens.
    addDependency(producer_stage, consumer);
  }

  return read;
}

void PipelineBuilder::receive(OpOperand& operand) {
  if (isAvailable(operand)) {
    return;
  }

  assert(canReceive(operand));
  auto consumer_stage = getStage(operand.getOwner());
  assert(consumer_stage);

  operand.set(send(cast<OpResult>(operand.get()), consumer_stage));
}

namespace {

enum class ForwardingResult : char {
  /// The result(s) can't be forwarded to the consumer(s).
  Failure = 0,
  /// Nothing needs to be forwarded.
  Skip,
  /// The result(s) can be forwarded to the consumer(s).
  Forward,
  /// The stage must be split before the result(s) can be forwarded.
  ///
  /// If forwarding the results would introduce a dependency of the producer on
  /// the consumer, the stage must be split to avoid the circular dependency.
  SplitAndForward,
};

/// Determines whether @p result can be forwarded when placed in @p stage .
[[nodiscard]] auto checkForwardingOf(
    StageOp stage, OpResult result, const PipelineBuilder::Allocator& allocator)
    -> ForwardingResult {
  auto status = ForwardingResult::Skip;

  for (auto* const user : result.getUsers()) {
    auto consumer_stage = getStage(stage->getBlock(), user);
    if (!consumer_stage) {
      // The user is not within the same pipeline.
      LDBG() << "  (FAILED) user of result #" << result.getResultNumber()
             << " isn't in the pipeline";
      return ForwardingResult::Failure;
    }
    if (consumer_stage == stage) {
      // This results stays within the same stage, so no forwarding needed.
      continue;
    }

    if (StageGraph::Node(stage).dependsOn(consumer_stage)) {
      // Forwarding this result would create a cyclic dependency.
      LDBG() << "  (WARN) detected dependency cycle between";
      LDBG() << "    consumer: " << OpWithFlags(consumer_stage, kSkipRegions);
      LDBG() << "    producer: " << OpWithFlags(stage, kSkipRegions);

      // We can break this cycle by creating a new stage. This stage will
      // copy the units of the old stage, and take its place in the map.
      status = ForwardingResult::SplitAndForward;
      break;
    }

    status = ForwardingResult::Forward;
  }

  if (status == ForwardingResult::Skip) {
    // We don't have to forward this result.
    return ForwardingResult::Skip;
  }

  if (!allocator.canAllocate(result)) {
    LDBG() << "  (FAILED) result #" << result.getResultNumber()
           << " can't be forwarded";
    return ForwardingResult::Failure;
  }

  return ForwardingResult::Forward;
};

/// Determines whether @p op can be forwarded when placed in @p stage .
///
/// @pre  All users of @p result are within the pipeline.
[[nodiscard]] auto checkForwardingOf(
    StageOp stage, Operation* op, SmallVectorImpl<OpResult>& forward,
    const PipelineBuilder::Allocator& allocator) -> ForwardingResult {
  auto status = ForwardingResult::Forward;
  for (auto result : op->getResults()) {
    switch (checkForwardingOf(stage, result, allocator)) {
      case ForwardingResult::Failure:
        return ForwardingResult::Failure;
      case ForwardingResult::Forward:
        break;
      case ForwardingResult::SplitAndForward:
        status = ForwardingResult::SplitAndForward;
        break;
      case ForwardingResult::Skip:
        continue;
    }

    forward.push_back(result);
  }

  return status;
}

}  // namespace

auto PipelineBuilder::insert(Operation* op, Placement placement)
    -> LogicalResult {
  assert(placement);

  IRRewriter rewriter(*this);

  LDBG() << "trying to insert";
  LDBG() << "    op: " << OpWithFlags(op, kSkipRegions);
  LDBG() << "  into: " << OpWithFlags(placement.stage, kSkipRegions);

  if (getStage(op)) {
    // This operation is already in the pipeline.
    LDBG() << "  (FAILED) already inside pipeline";
    return failure();
  }

  // Determine all the results that we will have to forward, checking for
  // dependency cycles in the process.
  SmallVector<OpResult> results_to_forward;
  switch (
      checkForwardingOf(placement.stage, op, results_to_forward, *allocator_)) {
    case ForwardingResult::Failure:
      return failure();
    case ForwardingResult::Forward:
    case ForwardingResult::Skip:
      break;
    case ForwardingResult::SplitAndForward: {
      // We can insert the op, but we have to break a dependency cycle. We can
      // do this by creating a new stage from the desired placement.
      auto old_stage = std::exchange(
          placement.stage, createStage(placement.stage.getApplicableUnitsAttr(),
                                       placement.stage.getLoc()));
      if (placement.erase_on_failure) {
        // This counts as a failure.
        rewriter.eraseOp(old_stage);
      }
      break;
    }
  }

  // This op can safely be moved into the pipeline.
  {
    op->remove();
    getOpBuilder(placement.stage).insert(op);
  }

  // Forward all results of this operation to their in-pipeline users. The
  // reads, writes and slots follow the users, so the order doesn't matter.
  for (auto result : results_to_forward) {
    for (auto& use : llvm::make_early_inc_range(result.getUses())) {
      auto stage = use.getOwner()->getParentOfType<StageOp>();
      assert(stage && "user outside of pipeline");
      receive(use);
    }
  }

  LDBG() << "  (SUCCESS) inserted";
  return success();
}

// Keeps the pending ops in a heap and always takes the latest one in program
// order. Pending ops haven't been moved yet, so their positions can be
// compared. Every op added to the heap is a producer of an op already visited,
// so it comes earlier in the program than that op. An op enters the heap at
// most once, so each op is visited once, after all of its pending users. An op
// that fails can't succeed later in the same call, so it isn't tried again.
void PipelineBuilder::insert(ArrayRef<Operation*> ops,
                             PlacementFn placement_fn) {
  assert(placement_fn);

  IRRewriter rewriter(*this);

  DenseSet<Operation*> seen;
  SmallVector<Operation*> work_list;
  for (auto* const op : ops) {
    if (seen.insert(op).second) {
      work_list.push_back(op);
    }
  }
  const auto later_first = [](Operation* lhs, Operation* rhs) -> bool {
    return isBeforeInProgram(lhs, rhs);
  };
  std::make_heap(work_list.begin(), work_list.end(), later_first);
  while (!work_list.empty()) {
    std::pop_heap(work_list.begin(), work_list.end(), later_first);
    auto* const op = work_list.pop_back_val();

    const auto placement = placement_fn(*this, op);
    if (!placement || failed(insert(op, placement))) {
      if (placement && placement.erase_on_failure) {
        // A later placement for the same units must not find the erased stage.
        auto stage = placement.stage;
        if (auto units = stage.getApplicableUnitsAttr();
            units && units_to_stage_.lookup(units) == stage) {
          units_to_stage_.erase(units);
        }
        rewriter.eraseOp(stage);
      }
      continue;
    }

    // Consider all the inserted ops producers for insertion into the pipeline.
    for (auto needs : op->getOperands()) {
      if (const auto wants = dyn_cast<OpResult>(needs);
          wants && seen.insert(wants.getOwner()).second) {
        LDBG() << "inserted op wants "
               << OpWithFlags(wants.getOwner(), kSkipRegions);
        work_list.push_back(wants.getOwner());
        std::push_heap(work_list.begin(), work_list.end(), later_first);
      }
    }
  }
}

PipelineBuilder::PipelineBuilder(PipelineOp pipeline, Allocator* allocator,
                                 OpBuilder::Listener* listener)
    : ImplicitLocOpBuilder(pipeline.getLoc(), pipeline.getContext(), listener),
      private_builder_(pipeline, std::nullopt, listener),
      allocator_(allocator != nullptr ? allocator : &Allocator::getDefault()) {
  OpBuilder::setInsertionPointToEnd(pipeline.getBody());
}
