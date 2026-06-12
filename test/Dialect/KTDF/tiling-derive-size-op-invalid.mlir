// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// Negative: zero [iv : tile_size] pairs must be rejected by the verifier.

func.func @tiling.derive_size_empty_pairs(%N: index) {
  // expected-error@+1 {{must have at least one [iv : tile_size] pair}}
  %x = ktdf.tiling.derive_size total_size = %N : index
  return
}
