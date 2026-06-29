//===-- KTDFArchTypes.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// KTDFArchDialect
//===----------------------------------------------------------------------===//

void KTDFArchDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.cpp.inc"

auto mlir::ktdf_arch::parseShortType(OpAsmParser& parser, Type& type)
    -> ParseResult {
  if (auto result = parser.parseOptionalType(type); result.has_value()) {
    return result.value();
  }

  const auto loc = parser.getCurrentLocation();
  StringRef keyword{};
  if (const auto maybe_short = generatedTypeParser(parser, &keyword, type);
      maybe_short.has_value()) {
    if (maybe_short.value()) {
      return parser.emitError(loc, "expected 'ktdf_arch' type mnemonic");
    }

    return success();
  }

  return parser.emitError(loc, "expected 'ktdf_arch' type mnemonic, got '")
         << keyword << "'";
}

void mlir::ktdf_arch::printShortType(OpAsmPrinter& printer, Operation* /*op*/,
                                     Type type) {
  if (succeeded(generatedTypePrinter(type, printer))) {
    return;
  }

  assert(!isa<BuiltinDialect>(type.getDialect()));
  printer << type;
}
