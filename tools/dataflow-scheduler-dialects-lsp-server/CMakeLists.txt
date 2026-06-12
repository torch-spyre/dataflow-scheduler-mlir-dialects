# Get the names of all relevant exported DataflowSchedulerDialects targets.
get_property(_dialect_libs GLOBAL PROPERTY ${PROJECT_NAME}_DIALECT_LIBS)
get_property(_extension_libs GLOBAL PROPERTY ${PROJECT_NAME}_EXTENSION_LIBS)

add_dataflow_scheduler_dialects_tool(dataflow-scheduler-dialects-lsp-server 
  dataflow-scheduler-dialects-lsp-server.cpp

  EXPORT_NAME lsp-server
)
target_link_libraries(dataflow-scheduler-dialects-lsp-server PRIVATE
  ${_dialect_libs}
  ${_extension_libs}
)

llvm_update_compile_flags(dataflow-scheduler-dialects-lsp-server)
mlir_target_link_libraries(dataflow-scheduler-dialects-lsp-server PRIVATE
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)
