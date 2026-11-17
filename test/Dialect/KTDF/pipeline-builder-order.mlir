// RUN: dataflow-scheduler-dialects-opt -allow-unregistered-dialect --test-ktdf-pipeline-builder %s | FileCheck %s

// The PipelineBuilder keeps the program order of the ops it moves into a stage.
// Each read goes before the first op of its stage that uses the value, and the
// writes from one stage to another are in the same order as their reads. This
// does not depend on the order in which values are sent.

// Two loads feed one compute with two results, each consumed by its own store.
// CHECK-LABEL: func.func @two_inputs_two_results
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private

// The loads keep their program order, and so do their writes.
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}

// Reads for the operands of one op follow the operand order.
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        %[[C:.+]]:2 = "test.compute"(%[[RA]], %[[RB]])
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]]#0, %[[P]]#[[C0_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]]#1, %[[P]]#[[C1_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["compute"]}

// Each read sits right before its store.
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[R0:.+]] = ktdf.read_from_fifo %[[P]]#[[C0_SLOT]]
// CHECK-NEXT:        "test.store"(%[[R0]]) {test.name = "x"
// CHECK-NEXT:        %[[R1:.+]] = ktdf.read_from_fifo %[[P]]#[[C1_SLOT]]
// CHECK-NEXT:        "test.store"(%[[R1]]) {test.name = "y"
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @two_inputs_two_results() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  %c:2 = "test.compute"(%a, %b) {test.units = "compute"} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  "test.store"(%c#0) {test.name = "x", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  "test.store"(%c#1) {test.name = "y", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  return
}

// A value that reaches the compute through a relay is inserted after the direct
// load. The relay's stage is still moved in front of the stage that uses it, so
// the stages stay in topological order.
// CHECK-LABEL: func.func @relay
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        %[[R:.+]] = "test.relay"(%[[RB]])
// CHECK-NEXT:        ktdf.write_to_fifo %[[R]], %[[P]]#[[R_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["relay"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        %[[RR:.+]] = ktdf.read_from_fifo %[[P]]#[[R_SLOT]]
// CHECK-NEXT:        "test.compute"(%[[RA]], %[[RR]])
// CHECK-NEXT:      } {applicable_units = ["compute"]}
func.func @relay() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  %r = "test.relay"(%b) {test.units = "relay"} : (tensor<4xf32>) -> tensor<4xf32>
  "test.compute"(%a, %r) {test.root, test.units = "compute"} : (tensor<4xf32>, tensor<4xf32>) -> ()
  return
}

// The results of one op are used in the reverse order. The writes follow the
// users, not the result numbers.
// CHECK-LABEL: func.func @results_cross_over
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[C:.+]]:2 = "test.compute"()
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]]#1, %[[P]]#[[C1_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]]#0, %[[P]]#[[C0_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["compute"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[R1:.+]] = ktdf.read_from_fifo %[[P]]#[[C1_SLOT]]
// CHECK-NEXT:        "test.store"(%[[R1]]) {test.name = "x"
// CHECK-NEXT:        %[[R0:.+]] = ktdf.read_from_fifo %[[P]]#[[C0_SLOT]]
// CHECK-NEXT:        "test.store"(%[[R0]]) {test.name = "y"
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @results_cross_over() {
  %c:2 = "test.compute"() {test.units = "compute"} : () -> (tensor<4xf32>, tensor<4xf32>)
  "test.store"(%c#1) {test.name = "x", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  "test.store"(%c#0) {test.name = "y", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  return
}

// Two producers in one stage are used in the reverse order in another stage.
// The ops keep their program order, and the writes follow the users.
// CHECK-LABEL: func.func @producers_cross_over
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RB]]) {test.name = "x"
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RA]]) {test.name = "y"
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @producers_cross_over() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  "test.store"(%b) {test.name = "x", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  "test.store"(%a) {test.name = "y", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  return
}

// A user nested in an `scf.if` gets its read at the top level, before the
// `scf.if`, so that the read is unconditional like its write.
// CHECK-LABEL: func.func @user_in_if
// CHECK-SAME:      (%[[COND:.+]]: i1)
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        scf.if %[[COND]] {
// CHECK-NEXT:          "test.store"(%[[RA]]) {test.name = "x"
// CHECK-NEXT:        } {test.root
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RB]]) {test.name = "y"
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @user_in_if(%cond: i1) {
  %a = "test.load"() {test.name = "a", test.root, test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  scf.if %cond {
    "test.store"(%a) {test.name = "x"} : (tensor<4xf32>) -> ()
  } {test.root, test.units = "store"}
  "test.store"(%b) {test.name = "y", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  return
}

// Users nested in an `scf.for` share one read, before the loop.
// CHECK-LABEL: func.func @user_in_for
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          "test.store"(%[[RA]]) {test.name = "x"
// CHECK-NEXT:          "test.store"(%[[RA]]) {test.name = "y"
// CHECK-NEXT:        } {test.root
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @user_in_for(%lb: index, %ub: index, %step: index) {
  %a = "test.load"() {test.name = "a", test.root, test.units = "load"} : () -> tensor<4xf32>
  scf.for %i = %lb to %ub step %step {
    "test.store"(%a) {test.name = "x"} : (tensor<4xf32>) -> ()
    "test.store"(%a) {test.name = "y"} : (tensor<4xf32>) -> ()
  } {test.root, test.units = "store"}
  return
}

// A value used in two consumer stages gets one FIFO per stage, and two users in
// the same stage share one read. The writes to different stages are not ordered
// against each other.
// CHECK-LABEL: func.func @two_consumer_stages
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-DAG:         ktdf.write_to_fifo %[[A]], %[[P]]#[[X_SLOT:[0-9]+]] : {{.*}}"load" -> "store_x"
// CHECK-DAG:         ktdf.write_to_fifo %[[A]], %[[P]]#[[Y_SLOT:[0-9]+]] : {{.*}}"load" -> "store_y"
// CHECK:           } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RX:.+]] = ktdf.read_from_fifo %[[P]]#[[X_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RX]]) {test.name = "x"
// CHECK-NEXT:      } {applicable_units = ["store_x"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RY:.+]] = ktdf.read_from_fifo %[[P]]#[[Y_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RY]]) {test.name = "y"
// CHECK-NEXT:        "test.store"(%[[RY]]) {test.name = "z"
// CHECK-NEXT:      } {applicable_units = ["store_y"]}
func.func @two_consumer_stages() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  "test.store"(%a) {test.name = "x", test.root, test.units = "store_x"} : (tensor<4xf32>) -> ()
  "test.store"(%a) {test.name = "y", test.root, test.units = "store_y"} : (tensor<4xf32>) -> ()
  "test.store"(%a) {test.name = "z", test.root, test.units = "store_y"} : (tensor<4xf32>) -> ()
  return
}

// Several roots inserted in one call come out in program order in each stage.
// CHECK-LABEL: func.func @roots_in_program_order
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        %[[C:.+]] = "test.load"() {test.name = "c"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]], %[[P]]#[[C_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RA]]) {test.name = "x"
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RB]]) {test.name = "y"
// CHECK-NEXT:        %[[RC:.+]] = ktdf.read_from_fifo %[[P]]#[[C_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RC]]) {test.name = "z"
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @roots_in_program_order() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  %c = "test.load"() {test.name = "c", test.units = "load"} : () -> tensor<4xf32>
  "test.store"(%a) {test.name = "x", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  "test.store"(%b) {test.name = "y", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  "test.store"(%c) {test.name = "z", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  return
}

// Writes to different stages are not ordered against each other, so they keep
// the program order of their values.
// CHECK-LABEL: func.func @writes_to_two_stages
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]] : {{.*}}"load" -> "store_x"
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]] : {{.*}}"load" -> "store_y"
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RA]]) {test.name = "x"
// CHECK-NEXT:      } {applicable_units = ["store_x"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        "test.store"(%[[RB]]) {test.name = "y"
// CHECK-NEXT:      } {applicable_units = ["store_y"]}
func.func @writes_to_two_stages() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  "test.store"(%a) {test.name = "x", test.root, test.units = "store_x"} : (tensor<4xf32>) -> ()
  "test.store"(%b) {test.name = "y", test.root, test.units = "store_y"} : (tensor<4xf32>) -> ()
  return
}
