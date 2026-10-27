// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt %s | FileCheck %s

// CHECK-DAG: #exec = {kind = "exec"}
// CHECK-DAG: #mem = {kind = "mem"}
// CHECK-DAG: #mem1 = {kind = "mem", reserved = 128 : i64, size = 1024 : i64}
// CHECK-DAG: #switch = {kind = "switch"}
// CHECK-DAG: #group = {kind = "group"}

// CHECK-LABEL: @my_device
ktdf_arch.device @my_device {
  // CHECK: exec_unit @exec #exec
  exec_unit @exec {kind = "exec"}
  // CHECK: memory @mem #mem
  memory @mem {kind = "mem"}
  // CHECK: memory @reserved_mem #mem1
  memory @reserved_mem {kind = "mem", size = 1024, reserved = 128}
  // CHECK: switch[3] @switch #switch
  switch[3] @switch {kind = "switch"}
  // CHECK: group @group #group share()
  group @group {kind = "group"} share() {}
}
