// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// One past the largest right shift of a 32-bit element.
func.func @shift_right_out_of_range(%x: vector<32xi32>) -> vector<32xi32> {
  // expected-error @+1 {{amount 32 is out of range [-31, 31] for the element type}}
  %0 = vectorchain.shift %x {amount = 32 : i32} : vector<32xi32>, vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// A left shift by the full width of a 32-bit element.
func.func @shift_left_full_width(%x: vector<32xi32>) -> vector<32xi32> {
  // expected-error @+1 {{amount -32 is out of range [-31, 31] for the element type}}
  %0 = vectorchain.shift %x {amount = -32 : i32} : vector<32xi32>, vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// One past the largest left shift of a 16-bit element.
func.func @shift_left_out_of_range(%x: vector<64xi16>) -> vector<64xi16> {
  // expected-error @+1 {{amount -16 is out of range [-15, 15] for the element type}}
  %0 = vectorchain.shift %x {amount = -16 : i32} : vector<64xi16>, vector<64xi16>
  return %0 : vector<64xi16>
}

// -----

// A shift moves bits, so a float element has nothing to shift.
func.func @shift_float(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{requires an integer element type}}
  %0 = vectorchain.shift %x {amount = 1 : i32} : vector<32xf32>, vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @shift_type_mismatch(%x: vector<32xi32>) -> vector<64xi16> {
  // expected-error @+1 {{result type does not match input type}}
  %0 = vectorchain.shift %x {amount = 1 : i32} : vector<32xi32>, vector<64xi16>
  return %0 : vector<64xi16>
}
