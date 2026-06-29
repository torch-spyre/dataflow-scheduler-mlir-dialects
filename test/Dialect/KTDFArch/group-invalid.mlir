// RUN: dataflow-scheduler-dialects-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

ktdf_arch.device @sibling_redefines_id {
  group share() {
    // expected-note@+1 {{previous definition is here}}
    exec_unit @a
    // expected-error@+1 {{resource with id "a" redefined}}
    exec_unit @a
  }
}

// -----

ktdf_arch.device @cousin_redefines_id {
  group share() {
    group share() {
      // expected-note@+1 {{previous definition is here}}
      exec_unit @a
    }
    // expected-error@+1 {{resource with id "a" redefined}}
    exec_unit @a
  }
}

// -----

ktdf_arch.device @unexpected_op {
  // expected-error@+1 {{expects child ops to be resources}}
  group share() {
    // expected-note@+1 {{unexpected child is here}}
    "dialect.op"() : () -> ()
  }
}

// -----

ktdf_arch.device @uncaptured_operand {
  %c = exec_unit
  group share() {
    // expected-error@+1 {{undeclared SSA value name}}
    "dialect.op"(%c) : (!ktdf_arch.exec_unit) -> ()
  }
}

// -----

ktdf_arch.device @invalid_yield {
  // expected-error@+1 {{region branch point has 0 operands, but region successor needs 1 inputs}}
  group share() {
    // expected-note@+1 {{region branch point}}
    yield
  } -> exec_unit
}
