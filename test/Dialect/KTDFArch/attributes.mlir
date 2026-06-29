// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt --allow-unregistered-dialect

func.func private @simd_lanes_attr() attributes { lanes = #ktdf_arch.map<f32 = 4, f16 = 8> }
