// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// Negative: modulo size < 1 must be rejected by the verifier.

func.func @pipeline_modulo_too_small() {
  // expected-error@+1 {{minimum value is 1}}
  ktdf.pipeline modulo(size: 0) {
    ktdf.stage depends_in(none) depends_out(none) {
      %c0 = arith.constant 0 : index
    }
  }
  return
}
