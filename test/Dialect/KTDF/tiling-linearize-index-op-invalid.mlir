// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// Negative: zero [iv : stride] pairs must be rejected by the verifier.

func.func @tiling_linearize_index_empty_pairs() {
  // expected-error@+1 {{must have at least one [iv : stride] pair}}
  %idx = ktdf.tiling.linearize_index : index
  return
}
