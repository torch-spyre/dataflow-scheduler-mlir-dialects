add_dataflow_scheduler_dialects_dialect(KTDF ktdf)
add_dataflow_scheduler_dialects_doc(KTDF KTDFDialect Dialects/ -gen-dialect-doc -dialect=ktdf)

set(LLVM_TARGET_DEFINITIONS KTDFEnums.td)
mlir_tablegen(KTDFEnums.h.inc -gen-enum-decls)
mlir_tablegen(KTDFEnums.cpp.inc -gen-enum-defs)
add_dataflow_scheduler_dialects_tablegen_target(DataflowSchedulerDialectsKTDFEnumsIncGen)
add_dependencies(DataflowSchedulerDialectsKTDFIncGen DataflowSchedulerDialectsKTDFEnumsIncGen)
