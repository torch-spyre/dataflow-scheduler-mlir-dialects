//===-- VectorChainEnums.td ---------------------------------*- tablegen -*-==//
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
// This file defines the enumerations in the vectorchain dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAINENUMS_TD
#define DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAINENUMS_TD

include "mlir/IR/EnumAttr.td"

def VectorChainBinaryAnd : I32EnumAttrCase<"and0", 0>;
def VectorChainBinaryOr : I32EnumAttrCase<"or0", 1>;
def VectorChainBinaryXnor : I32EnumAttrCase<"xnor", 2>;
def VectorChainBinaryAndNot : I32EnumAttrCase<"and_not", 3>;
def VectorChainBinaryMin : I32EnumAttrCase<"min", 4>;
def VectorChainBinaryMax : I32EnumAttrCase<"max", 5>;
def VectorChainBinaryAbsMin : I32EnumAttrCase<"abs_min", 6>;
def VectorChainBinaryAbsMax : I32EnumAttrCase<"abs_max", 7>;
def VectorChainBinaryAdd : I32EnumAttrCase<"add", 8>;
def VectorChainBinaryMul : I32EnumAttrCase<"mul", 9>;
def VectorChainBinarySub : I32EnumAttrCase<"sub", 10>;
def VectorChainBinaryMulDiv2 : I32EnumAttrCase<"mul_div2", 11>;

def VectorChainBinaryOperator : I32EnumAttr<"VectorChainBinaryOperator",
    "built-in operators supported by VectorChain binary ops.",
    [
      VectorChainBinaryAnd,
      VectorChainBinaryOr,
      VectorChainBinaryXnor,
      VectorChainBinaryAndNot,
      VectorChainBinaryMin,
      VectorChainBinaryMax,
      VectorChainBinaryAbsMin,
      VectorChainBinaryAbsMax,
      VectorChainBinaryAdd,
      VectorChainBinaryMul,
      VectorChainBinarySub,
      VectorChainBinaryMulDiv2
    ]> {
      let genSpecializedAttr = 0;
      let cppNamespace = "::mlir::vectorchain";
    }

#endif // DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAINENUMS_TD
