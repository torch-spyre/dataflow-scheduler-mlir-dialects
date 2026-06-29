add_dataflow_scheduler_dialects_dialect(Agen agen)
add_dataflow_scheduler_dialects_doc(Agen AgenDialect Dialects/ -gen-dialect-doc -dialect=agen)

set(LLVM_TARGET_DEFINITIONS AgenEnums.td)
mlir_tablegen(AgenEnums.h.inc -gen-enum-decls)
mlir_tablegen(AgenEnums.cpp.inc -gen-enum-defs)
add_dataflow_scheduler_dialects_tablegen_target(DataflowSchedulerDialectsAgenEnumsIncGen)
add_dependencies(DataflowSchedulerDialectsAgenIncGen DataflowSchedulerDialectsAgenEnumsIncGen)
