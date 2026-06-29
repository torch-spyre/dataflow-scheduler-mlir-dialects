add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsDataflowDialect
    Dataflow.cpp
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowSchedulerDialects_SOURCE_DIR}/include/dataflow-scheduler/Dialect/Dataflow
    ${DataflowSchedulerDialects_BINARY_DIR}/include/dataflow-scheduler/Dialect/Dataflow

  GROUP DIALECT
  EXPORT_NAME DataflowDialect

  DEPENDS
    DataflowSchedulerDialectsDataflowIncGen
)

mlir_target_link_libraries(DataflowSchedulerDialectsDataflowDialect PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSCFDialect
)
