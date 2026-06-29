//===-- Agen.td ---------------------------------------------*- tablegen -*-==//
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
// This file defines the agen dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_AGEN_AGEN_TD
#define DATAFLOW_SCHEDULER_DIALECT_AGEN_AGEN_TD

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/Agen/AgenEnums.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Agen_Dialect : Dialect {
  let name = "agen";
  let cppNamespace = "::mlir::agen";

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "arith::ArithDialect",
    "dataflow::DataflowDialect",
    "vectorchain::VectorChainDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def AgenRoutingDirectionAttr : EnumAttr<Agen_Dialect, AgenRoutingDirection, "direction">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// Base class for Agen dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Agen_Op<string mnemonic, list<Trait> traits = []> :
    Op<Agen_Dialect, mnemonic, traits> {
}

// Require regions to have sentient.yield.
def ImplicitAgenTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;

def YieldOp : Agen_Op<"yield", [NoMemoryEffect, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "agen.yield yields zero or more SSA values to parent op."
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Agen_VectorLoadOp : Agen_Op<"vector_load", [
    DeclareOpInterfaceMethods<AffineReadOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable,
    AttrSizedOperandSegments]> {
  let summary = "agen vector load operation";
  let description = [{
    Loads a vector of data from a memory view (typically produced by
    `dataflow.get_logical_memory_view`).

    The index subscripts on `$mem_ref` give the starting address of the load.
    `$load_set` is an affine set that dictates which elements (relative to the
    starting address) are gathered into the vector, and `$load_order` is an
    affine map that dictates how those elements are packed into the result
    vector. The load is contiguous when the set walks the innermost memory
    dimension; selecting an outer dimension instead yields a strided load.

    `$multicast_info`, when present, carries routing information for transfers
    that fan the loaded vector out to multiple consumers.

    Example:
    ```mlir
    %data = agen.vector_load %l1_view[0, 0]
      {load_order = affine_map<(d0, d1) -> (d1, d0)>,
       load_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 == 0)>}
      : memref<8x16xi8>, vector<8xi8>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$mem_ref,
    OptionalAttr<StrAttr>:$dbg_name,
    AffineMapAttr:$affine_map,
    Variadic<Index>:$map_operands,
    Builtin_IntegerSetAttr:$load_set,
    AffineMapAttr:$load_order,
    Optional<Index>:$multicast_info);

  let results = (outs AnyVectorOfAnyRank:$result);

  let builders = [
    /// Builds an agen vector load op with the specified map and operands.
    OpBuilder<(ins 
        "VectorType":$resultType, "Value":$mem_ref, "StringAttr":$dbg_name, 
        "AffineMap":$map, "ValueRange":$map_operands, "IntegerSet":$load_set, 
        "AffineMap":$load_order, CArg<"Value", "{}">:$multicast_info), [{
      build($_builder, $_state, resultType, mem_ref, dbg_name, map, map_operands, 
            IntegerSetAttr::get(load_set), load_order, multicast_info);
    }]>
  ];

  let assemblyFormat = [{
    $mem_ref``custom<AffineMapOfSSAIds>($affine_map, $map_operands)
    (`multicast_info` `=` $multicast_info^)?
    attr-dict `:` type($mem_ref) `,` type($result)
  }];

  let extraClassDeclaration =  [{
    //===------------------------------------------------------------------===//
    // AffineMapAccessInterface
    //===------------------------------------------------------------------===//

    [[nodiscard]] auto getMapAttrStrName() -> StringRef {
      return getAffineMapAttrName();
    }
  }];
}

def Agen_VectorStoreOp : Agen_Op<"vector_store", [
    DeclareOpInterfaceMethods<AffineWriteOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable]> {
  let summary = "agen vector store operation";
  let description = [{
    Stores a vector of data into a memory view (typically produced by
    `dataflow.get_logical_memory_view`). This is the counterpart of
    `agen.vector_load` and carries the analogous fields.

    The index subscripts on `$mem_ref` give the starting address of the store.
    `$store_set` is an affine set that dictates which elements (relative to the
    starting address) are written, and `$store_order` is an affine map that
    dictates the order in which lanes of `$value_to_store` map onto those
    elements.

    Note that the starting address may change across iterations, as arises with
    double-buffering or streaming transfers.

    Example:
    ```mlir
    agen.vector_store %data, %l1_view[0, 0]
      {store_order = affine_map<(d0, d1) -> (d1, d0)>,
       store_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 == 0)>}
      : memref<8x16xi8>, vector<8xi8>
    ```
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$value_to_store,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$mem_ref,
    OptionalAttr<StrAttr>:$dbg_name,
    AffineMapAttr:$affine_map,
    Variadic<Index>:$map_operands,
    Builtin_IntegerSetAttr:$store_set,
    AffineMapAttr:$store_order);

  let builders = [
    OpBuilder<(ins 
        "Value":$value_to_store, "Value":$mem_ref, "StringAttr":$dbg_name, "AffineMap":$map, 
        "ValueRange":$map_operands, "IntegerSet":$store_set, "AffineMap":$store_order), [{
      build($_builder, $_state, value_to_store, mem_ref, dbg_name, map, map_operands, 
            IntegerSetAttr::get(store_set), store_order);
    }]>,
    OpBuilder<(ins 
        "Value":$value_to_store, "Value":$mem_ref, "mlir::StringAttr":$dbg_name,
        "ValueRange":$indices, "IntegerSet":$store_set, "mlir::AffineMap":$store_order)>
  ];

  let assemblyFormat = [{
    $value_to_store `,` $mem_ref``custom<AffineMapOfSSAIds>($affine_map, $map_operands)
    attr-dict `:` type($mem_ref) `,` type($value_to_store)
  }];

  let extraClassDeclaration =  [{
    //===------------------------------------------------------------------===//
    // AffineMapAccessInterface
    //===------------------------------------------------------------------===//

    [[nodiscard]] auto getMapAttrStrName() -> StringRef {
      return getAffineMapAttrName();
    }
  }];
}

def Agen_CompositeLoadAndStoreOp : Agen_Op<"composite_load_and_store", [
    MemRefsNormalizable,
    ImplicitAgenTerminator,
    DeclareOpInterfaceMethods<AffineMapAccessInterface, ["getAffineMapAttrForMemRef"]>,
    AttrSizedOperandSegments]> {
  let summary = "agen composite load_and_store operation";
  let description = [{
    Captures a sequence of individual vector loads and stores: at each time
    step it loads from one memory address and stores into another. This is used
    to model core-to-memory or memory-to-core data transfers.

    The element dimension is described by `$load_set`/`$store_set` (which
    elements form each vector) and `$load_order`/`$store_order` (how those
    elements are packed), mirroring `agen.vector_load`/`agen.vector_store`.

    The time dimension is described by:
      * `$time_set` - the set of time steps involved in the composite operation.
      * `$time_order` - the order among those time steps.
      * `$load_time_addr_map`/`$store_time_addr_map` - the address offset from
        the base address for each time step.

    The single region is entered once per time step. Its induction argument (see
    `getLoadInductionVar`) carries the vector loaded at that time step, and the
    region body describes what to do with it (for example, sending it to another
    unit, or storing it via the terminating `agen.yield`). `$dir` and
    `$multicast_info`, when present, carry routing information for the transfer.

    Example (three time steps loading columns 0, 1, 2):
    ```mlir
    agen.composite_load_and_store src:%src[0, 0] dst:%dst[0, 0]
      time_symbols(), load_iv(%load:vector<8xi8>)
      {load_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 == 0)>,
       load_order = affine_map<(d0, d1) -> (d1, d0)>,
       time_set = affine_set<(t0) : (t0 >= 0, -t0 + 2 >= 0)>,
       time_order = affine_map<(t0) -> (t0)>,
       load_time_addr_map = affine_map<(t0) -> (0, t0)>, ...}
    {
      // region: consume %load, e.g. agen.yield
    } : memref<8x16xi8>, memref<8x16xi8>
    ```
  }];
  let arguments = (ins 
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$src_mem_ref,
    Arg<AnyMemRef, "the reference to store into", [MemWrite]>:$dst_mem_ref,
    OptionalAttr<StrAttr>:$dbg_name,
    AffineMapAttr:$src_map,
    Variadic<Index>:$src_operands,
    AffineMapAttr:$dst_map,
    Variadic<Index>:$dst_operands,
    Builtin_IntegerSetAttr:$load_set,
    AffineMapAttr:$load_order,
    Builtin_IntegerSetAttr:$store_set,
    AffineMapAttr:$store_order,
    Variadic<Index>:$time_symbols,
    Builtin_IntegerSetAttr:$time_set,
    AffineMapAttr:$time_order,
    AffineMapAttr:$load_time_addr_map,
    AffineMapAttr:$store_time_addr_map,
    OptionalAttr<AgenRoutingDirectionAttr>:$dir,
    Optional<Index>:$multicast_info);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins 
        "Value":$src_mem_ref, "Value":$dst_mem_ref, "StringAttr":$dbg_name, 
        "AffineMapAttr":$src_map, "ValueRange":$src_operands, "AffineMapAttr":$dst_map, 
        "ValueRange":$dst_operands, "IntegerSetAttr":$load_set, "AffineMapAttr":$load_order,  
        "IntegerSetAttr":$store_set, "AffineMapAttr":$store_order, "ValueRange":$time_symbols,
        "IntegerSetAttr":$time_set, "AffineMapAttr":$time_order, "AffineMapAttr":$load_time_addr_map, 
        "AffineMapAttr":$store_time_addr_map, "VectorType":$type,
        CArg<"AgenRoutingDirectionAttr", "nullptr">:$dir,
        CArg<"Value", "{}">:$multicast_info,
        CArg<"function_ref<void(OpBuilder &, Location, Value)>", "nullptr">:$body_builder)>,
    /// Builds a composite load_and_store op with the specified map and operands.
    OpBuilder<(ins 
        "Value":$src_mem_ref, "Value":$dst_mem_ref, "StringAttr":$dbg_name, 
        "AffineMap":$src_map, "ValueRange":$src_operands, "AffineMap":$dst_map, 
        "ValueRange":$dst_operands, "IntegerSet":$load_set, "AffineMap":$load_order,  
        "IntegerSet":$store_set, "AffineMap":$store_order, "ValueRange":$time_symbols,
        "IntegerSet":$time_set, "AffineMap":$time_order, "AffineMap":$load_time_addr_map, 
        "AffineMap":$store_time_addr_map, "VectorType":$type,
        CArg<"AgenRoutingDirectionAttr", "nullptr">:$dir,
        CArg<"Value", "{}">:$multicast_info,
        CArg<"function_ref<void(OpBuilder &, Location, Value)>", "nullptr">:$body_builder)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{    
    [[nodiscard]] auto getLoadInductionVar() -> TypedValue<VectorType> {
      return cast<TypedValue<VectorType>>(
        getRegion().getBlocks().begin()->getArgument(0));
    }
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_AGEN_AGEN_TD
