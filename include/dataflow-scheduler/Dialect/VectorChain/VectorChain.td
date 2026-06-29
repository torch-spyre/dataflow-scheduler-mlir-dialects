//===-- VectorChain.td -------------------------------------*- tablegen ----==//
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
// This file defines the vectorchain dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAIN_TD
#define DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAIN_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/VectorChain/VectorChainEnums.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def VectorChain_Dialect : Dialect {
  let name = "vectorchain";
  let cppNamespace = "::mlir::vectorchain";

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "dataflow::DataflowDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def VectorChainBinaryOperatorAttr : EnumAttr<VectorChain_Dialect, VectorChainBinaryOperator, "binary_operator">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// Base class for VectorChain dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class VectorChain_Op<string mnemonic, list<Trait> traits = []> :
    Op<VectorChain_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
}

def VectorChain_BinaryOp:VectorChain_Op<"binary",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Element-wise binary operation";
    let description = [{
    Takes two operand vectors, performs the element-wise operation named by the
    `binary_op` attribute (for example `multiply`, `and`, `or`, `min`), and
    reduces the result. `op_specific_map` is an affine map describing the
    operation-specific reduction/indexing. The optional `mask` selects which
    lanes of the result participate in the computation.

    ```mlir
    %res = vectorchain.binary %a, %b [%mask]
        {binary_op = #vectorchain<binary_operator multiply>,
         op_specific_map = affine_map<(i) -> (i)>}
        : vector<64xf16>, vector<64xf16>, vector<64xf16>
    ```
    }];
    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainBinaryOperatorAttr:$binary_op,
                         AffineMapAttr:$op_specific_map);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_ShuffleOp:VectorChain_Op<"shuffle",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Shuffle operation";
    let description = [{
    Constructs a vector by shuffling/selecting the elements of an input vector.

    Each element of `indices` refers to an element of the input vector. The
    `indices` pattern is repeated `repetition` times, so the product of the
    number of `indices` and `repetition` must equal the number of elements in
    the output vector. An index value of -1 indicates a padding element.

    ```mlir
    %splat = vectorchain.shuffle %data {indices = [0, 1], repetition = 64 : i32}
        : vector<2xi8>, vector<128xi8>
    ```
    }];

    let arguments = (ins AnyVectorOfAnyRank:$input,
                         Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         ArrayAttr:$indices,
                         I32Attr:$repetition);

    let results = (outs AnyVectorOfAnyRank:$output);
    let assemblyFormat = [{
          $input (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($input) `,` type(results)
    }];

    let hasVerifier = 1;
}

#endif // DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAIN_TD
