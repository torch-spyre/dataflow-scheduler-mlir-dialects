// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

func.func @bad_source_size_count(%i: index) {
  %A = memref.alloc() : memref<128x64xf16, "DDR">
  %l1 = memref.alloc() : memref<128x64xf16, "L1">
  // expected-error@+1 {{source size count (1) must match memref rank (2)}}
  "ktdf.data_transfer"(%A, %i, %l1, %i) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0>,
    static_source_sizes = array<i64: 64>,
    static_dest_sizes = array<i64: 128, 64>,
    source_map = affine_map<(d0) -> (d0, d0)>,
    dest_map = affine_map<(d0) -> (d0, d0)>
  }> : (memref<128x64xf16, "DDR">, index,
        memref<128x64xf16, "L1">, index) -> ()
  return
}

// -----

func.func @bad_dest_size_count(%i: index) {
  %l1 = memref.alloc() : memref<128x64xf16, "L1">
  %B = memref.alloc() : memref<128x64xf16, "DDR">
  // expected-error@+1 {{dest size count (3) must match memref rank (2)}}
  "ktdf.data_transfer"(%l1, %i, %B, %i) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0>,
    static_source_sizes = array<i64: 128, 64>,
    static_dest_sizes = array<i64: 1, 128, 64>,
    source_map = affine_map<(d0) -> (d0, d0)>,
    dest_map = affine_map<(d0) -> (d0, d0)>
  }> : (memref<128x64xf16, "L1">, index,
        memref<128x64xf16, "DDR">, index) -> ()
  return
}

// -----

func.func @fifo_to_memref_valid(%i: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
  %l1 = memref.alloc() : memref<64xf16, "L1">
  "ktdf.data_transfer"(%fifo, %l1, %i) <{
    operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 0>,
    static_source_sizes = array<i64: 64>,
    static_dest_sizes = array<i64: 64>,
    dest_map = affine_map<(d0) -> (d0)>
  }> : (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
        memref<64xf16, "L1">, index) -> ()
  return
}
