// RUN: dataflow-scheduler-dialects-opt -allow-unregistered-dialect %s | dataflow-scheduler-dialects-opt -allow-unregistered-dialect | FileCheck %s
// Round-tripping test to ensure pipeline and stage operations are registered properly

// CHECK-LABEL:   func.func @pipeline_with_no_dependencies() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @pipeline_with_single_stage() {
// CHECK-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]]) {
// CHECK-NEXT:         %[[VAL_0:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:         "test.op"(%[[VAL_0]]) : (!ktdf.token) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]]) depends_out(none) {
// CHECK-NEXT:         %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:         "test.op"(%[[VAL_1]]) : (!ktdf.token) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @pipeline_with_multiple_stages() {
// CHECK-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]]) {
// CHECK-NEXT:         %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]]) depends_out(%[[CREATE_TOKEN_1]]) {
// CHECK-NEXT:         %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_1]]) depends_out(none) {
// CHECK-NEXT:         %[[CONSTANT_2:.*]] = arith.constant 3 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @nested_pipeline_operations() {
// CHECK-NEXT:     %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_0]]) {
// CHECK-NEXT:         %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"l1lu-to-sfu", 1xf16>, !ktdf.fifo.slot<"l1lu-to-sfu", 4xf16>
// CHECK-NEXT:         "test.op"(%[[FIFO_0]]#0) : (!ktdf.fifo.slot<"l1lu-to-sfu", 1xf16>) -> ()
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[CREATE_TOKEN_1]]) {
// CHECK-NEXT:             %[[VAL_0:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:             "test.op"(%[[VAL_0]]) : (!ktdf.token) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[CREATE_TOKEN_0]]) depends_out(none) {
// CHECK-NEXT:         %[[VAL_1:.*]] = "test.op"() : () -> !ktdf.token
// CHECK-NEXT:         "test.op"(%[[VAL_1]]) : (!ktdf.token) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func @pipeline_with_no_dependencies() {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        %c = arith.constant 42 : i32
      }
    }
    return
  }

  func.func @pipeline_with_single_stage() {
    %token = ktdf.create_token : !ktdf.token
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(%token) {
        %t = "test.op"() : () -> !ktdf.token
        "test.op"(%t) : (!ktdf.token) -> ()
      }
      ktdf.stage depends_in(%token) depends_out(none) {
        %t = "test.op"() : () -> !ktdf.token
        "test.op"(%t) : (!ktdf.token) -> ()
      }
    }
    return
  }

  func.func @pipeline_with_multiple_stages() {
    %token_a = ktdf.create_token : !ktdf.token
    %token_b = ktdf.create_token : !ktdf.token
    
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(%token_a) {
        %0 = arith.constant 1 : i32
      }
      ktdf.stage depends_in(%token_a) depends_out(%token_b) {
        %1 = arith.constant 2 : i32
      }
      ktdf.stage depends_in(%token_b) depends_out(none) {
        %2 = arith.constant 3 : i32
      }
    }
    return
  }

  func.func @nested_pipeline_operations() {
    %token_outer = ktdf.create_token : !ktdf.token
    %token_inner = ktdf.create_token : !ktdf.token
    
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(%token_outer) {
        %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"l1lu-to-sfu", 1xf16>, !ktdf.fifo.slot<"l1lu-to-sfu", 4xf16>
        "test.op"(%slot0) : (!ktdf.fifo.slot<"l1lu-to-sfu", 1xf16>) -> ()
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(%token_inner) {
            %token = "test.op"() : () -> !ktdf.token
            "test.op"(%token) : (!ktdf.token) -> ()
          }
        }
      }
      ktdf.stage depends_in(%token_outer) depends_out(none) {
        %token = "test.op"() : () -> !ktdf.token
        "test.op"(%token) : (!ktdf.token) -> ()
      }
    }
    return
  }
}