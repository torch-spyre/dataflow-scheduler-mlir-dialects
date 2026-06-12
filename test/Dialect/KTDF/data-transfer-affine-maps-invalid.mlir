// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// Verifier: source_map result count must match source memref rank.

func.func @bad_source_map_rank(%i: index) {
  %A = memref.alloc() : memref<128x128xf16, "DDR">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"ddr-to-sfu", 64xf16>
  // expected-error@+1 {{source memref has rank 2 but source_map has 3 results}}
  "ktdf.data_transfer"(%A, %i, %fifo) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0>,
    static_source_sizes = array<i64: 1, 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<(d0) -> (d0, d0, d0)>
  }> : (memref<128x128xf16, "DDR">, index,
        !ktdf.fifo.slot<"ddr-to-sfu", 64xf16>) -> ()
  return
}

// -----

// Verifier: source_map cannot have symbols.
func.func @bad_source_map_symbols(%i: index, %n: index) {
  %A = memref.alloc() : memref<128x128xf16, "DDR">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"ddr-to-sfu", 64xf16>
  // expected-error@+1 {{source_map must have 0 symbols (dims-only); got 1}}
  "ktdf.data_transfer"(%A, %i, %n, %fifo) <{
    operandSegmentSizes = array<i32: 1, 2, 0, 1, 0, 0>,
    static_source_sizes = array<i64: 1, 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<(d0)[s0] -> (d0 + s0, d0)>
  }> : (memref<128x128xf16, "DDR">, index, index,
        !ktdf.fifo.slot<"ddr-to-sfu", 64xf16>) -> ()
  return
}

// -----

// Verifier: source_map must be absent for fifo source.
func.func @bad_fifo_with_map() {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"ddr-to-sfu", 64xf16>
  %l1 = memref.alloc() : memref<64xf16, "L1">
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{source_map must be absent for fifo source}}
  "ktdf.data_transfer"(%fifo, %l1, %c0) <{
    operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 0>,
    static_source_sizes = array<i64: 64>,
    static_dest_sizes = array<i64: 64>,
    source_map = affine_map<() -> ()>,
    dest_map = affine_map<(d0) -> (d0)>
  }> : (!ktdf.fifo.slot<"ddr-to-sfu", 64xf16>, memref<64xf16, "L1">,
        index) -> ()
  return
}
