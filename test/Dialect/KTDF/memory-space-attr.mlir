// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Round tripping test for memref string memory spaces used by the dialect.

// CHECK-LABEL: func.func @test_memory_space_attr
// CHECK: memref<4096xi32, "DDR">

func.func @test_memory_space_attr() {
  %1 = memref.alloc() : memref<4096xi32, "DDR">
  return
}