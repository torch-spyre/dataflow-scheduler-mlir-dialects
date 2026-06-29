//===-- Dataflow.td -----------------------------------------*- tablegen -*-==//
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
// This file defines the dataflow dialect attributes and operations.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOW_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOW_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/Dataflow/DataflowEnums.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::mlir::dataflow";
  
  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "arith::ArithDialect",
    // These are used by Utils.cpp, and may require loading.
    "affine::AffineDialect",
    "scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def DataflowRoutingDirectionAttr : EnumAttr<Dataflow_Dialect, DataflowRoutingDirection, "direction">; 

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// Base class for Dataflow dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dataflow_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dataflow_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
}

def Dataflow_GetUnitOp:Dataflow_Op<"get_unit",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Getting a handle to a global unit";
    let description = [{
    Returns an SSA handle corresponding to a hardware unit. The unit is
    identified by its attributes - typically `core`, `corelet`, `type`, and
    `name`. The handle is then used by other dataflow operations (such as
    `program_unit`, `send`, `receive`, and `get_logical_memory_view`) to refer
    to that unit.

    A `get_unit` may produce multiple results, one per program time step.

    ```mlir
    %l1lu = dataflow.get_unit {core = 0, corelet = 0, type = "L1LU", name = "L1-LU"} : index
    ```
    }];
    let arguments = (ins StrAttr:$name, StrAttr:$type);

    //Variadic has been added to represent multiple return values of GetUnit operation with each return
    //value corresponding to an instance of program time steps.
    let results = (outs Variadic<Index>:$units);
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      Value getUnit();
    }];
}

def Dataflow_ProgramUnitOp:Dataflow_Op<"program_unit",
  [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Definition of a unit";
    let description = [{
    Holds the program associated with one or more units. The region carries the
    body of work executed on the given unit(s) and is terminated by a
    `dataflow.return`. The optional `precision` attribute records the numeric
    precision of the unit's program.

    ```mlir
    dataflow.program_unit (%l1lu) { #region } : index
    ```

    Multiple units can be bound through iter-arg style operands, one per program
    time step:

    ```mlir
    dataflow.program_unit iter_arg : %unit -> (%l1lu0, %l1lu1) { #region } : index, index
    ```
    }];

    let arguments = (ins Variadic<Index>:$units, OptionalAttr<StrAttr>:$precision);
    let regions = (region SizedRegion<1>:$region);
    let skipDefaultBuilders = 1;
    let builders = [
    OpBuilder<(ins "ValueRange":$units,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilder)>,
      OpBuilder<(ins "ValueRange":$units, "StringAttr":$precision,
            CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                 "nullptr">:$bodyBuilder)>
    ];

    let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;
        static StringRef getPrecisionAttrStrName() { return "precision"; }
    }];

    let hasCustomAssemblyFormat = 1;
}

def Dataflow_ReturnOp : Dataflow_Op<"return", [Terminator]> {
  let summary = [{
    The `return` operation terminates the definition.
  }];
  let description = [{
    The `return` operation is the terminator of a `dataflow.program_unit`
    region. It marks the end of the unit definition and carries no operands.
    It is the implicit terminator for that region, so it is normally created
    automatically rather than written by hand.
  }];
}


def Dataflow_CreateGroupOp:Dataflow_Op<"create_group",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Creates a group of unit IDs";
    let description = [{
    Creates a logical group of units from the given unit handles and returns a
    single group handle. The group can then be used wherever a collective of
    units is addressed (for example as a multicast destination).

    ```mlir
    %group = dataflow.create_group (%l1lu0 : index, %l1lu1 : index) : index
    ```
    }];

    let arguments = (ins Variadic<Index>:$unit_ids);
    let results = (outs Index:$group_id);
    let assemblyFormat = [{
       `(` ($unit_ids^ `:` type($unit_ids))? `)`  attr-dict `:` type(results)
    }];
}

def Dataflow_SyncSendOp:Dataflow_Op<"sync_send",
  []> {
    let summary = "Send synchronization signal for explicit synchronization";
    let description = [{
    Sends a synchronization signal to another unit, used for explicit
    synchronization between units. The optional
    `wait_immediately_for_async_transfers` attribute controls whether the sender
    waits for outstanding asynchronous transfers before signalling.

    ```mlir
    dataflow.sync_send %l1su0 {wait_immediately_for_async_transfers = false} : index
    ```
    }];
    let arguments = (ins Index:$to_unit, OptionalAttr<StrAttr>:$dbgName,
                         OptionalAttr<BoolAttr>:$wait_immediately_for_async_transfers);
    let assemblyFormat = [{
         $to_unit attr-dict `:` type($to_unit)
    }];
}

def Dataflow_SyncRecvOp:Dataflow_Op<"sync_recv",
  []> {
    let summary = "Receive synchronization signal for explicit synchronization";
    let description = [{
    Receives a synchronization signal from another unit, used for explicit
    synchronization between units. This is a blocking call: it does not return
    until the matching signal has been received.

    ```mlir
    dataflow.sync_recv %l1su0 : index
    ```
    }];
    let arguments = (ins Index:$from_unit,
                         OptionalAttr<StrAttr>:$dbgName);
    let assemblyFormat = [{
         $from_unit  attr-dict `:` type($from_unit)
    }];
}

def Dataflow_GetLogicalMemoryViewOp:Dataflow_Op<"get_logical_memory_view",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to get a logical view of the memory";
    let description = [{
    Views a hardware memory region, addressed linearly from `$start_address`, as
    a multi-dimensional memref. The `layout_map` affine map describes how the
    multi-dimensional indices of the view map back onto the 1-D address space.

    Addresses in Dataflow IR are in *element* granularity, not bytes. For
    example, with `layout_map = affine_map<(i, j) -> (i + j * 8)>`, index
    `(0, 1)` refers to element `0 + 1 * 8 = 8` of the underlying 1-D region.

    ```mlir
    %l1_view = dataflow.get_logical_memory_view (%l1_mem, %start_address)
        {layout_map = affine_map<(i, j) -> (i + j * 8)>}
        : index, index, memref<8x16xi8>
    ```
    }];
    let arguments = (ins Index:$from_unit, Index:$start_address, AffineMapAttr:$layout_map);
    let results = (outs AnyTypeOf<[AnyMemRef]>:$data);
    let assemblyFormat = [{
         $from_unit `,` $start_address  attr-dict `:` type($from_unit) `,` type($start_address) `,` type(results)
    }];
}

def Dataflow_SendOp:Dataflow_Op<"send",
  []> {
    let summary = "Send a vector of data to another unit";
    let description = [{
    Sends a vector of data to another unit. The optional `dir` attribute records
    the routing direction of the transfer.

    ```mlir
    dataflow.send %l0su, %data : index, vector<128xi8>
    ```
    }];

    let arguments = (ins Index:$to_unit,
                     AnyTypeOf<[AnyVectorOfAnyRank]>:$send_data,
                     OptionalAttr<DataflowRoutingDirectionAttr>:$dir,
                     OptionalAttr<StrAttr>:$dbgName
                     );
    let assemblyFormat = [{
         $to_unit `,` $send_data attr-dict `:` type($send_data)
    }];
}

def Dataflow_ReceiveOp:Dataflow_Op<"receive",
  []> {
    let summary = "Receive a vector of data from another unit";
    let description = [{
    Receives a vector of data from another unit and returns it as the result.

    ```mlir
    %data = dataflow.receive %l1lu : vector<128xi8>
    ```
    }];
    let arguments = (ins Index:$from_unit,
                         OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyTypeOf<[AnyVectorOfAnyRank]>:$data);

    let assemblyFormat = [{
         $from_unit attr-dict `:` type(results)
    }];
}


#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOW_TD
