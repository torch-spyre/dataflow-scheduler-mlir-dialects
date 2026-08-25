add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFArchTransforms
    ApplyPatterns.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDFArch/Transforms
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDFArch/Transforms

  GROUP DIALECT
  EXPORT_NAME KTDFArchTransforms

  DEPENDS
    DataflowSchedulerDialectsKTDFArchTransformsPassesIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowSchedulerDialects::AgenDialect
    DataflowSchedulerDialects::VectorChainDialect

  LINK_MLIR_LIBS
    MLIRFuncDialect
    MLIRIR
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRRewrite
    MLIRPDLDialect
    MLIRPDLInterpDialect
)
