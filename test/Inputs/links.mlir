ktdf_arch.device @my_device attributes {version = 1} {
  %a = group share() {
    %a = exec_unit @a {
      expect_incoming = [], 
      expect_outgoing = [@a_to_b]
    }
    yield %a
  } -> exec_unit

  %b = exec_unit @b {
    expect_incoming = [@a_to_b, @c_to_b], 
    expect_outgoing = [@b_to_c],
    expect_to = #ktdf_arch.map<@c = [@b_to_c]>
  }

  datapath @a_to_b %a to %b : exec_unit, exec_unit

  %d = memory @d {
    kind = "d",
    expect_incoming = [@c_to_d_1, @c_to_d_2], 
    expect_outgoing = []
  }

  %c = group share(%d) {
    %c = exec_unit @c {
      expect_incoming = [@b_to_c], 
      expect_outgoing = [@c_to_b, @c_to_d_1, @c_to_d_2],
      expect_to = #ktdf_arch.map<@b = [@c_to_b], @d = [@c_to_d_1, @c_to_d_2]>
    }

    datapath @c_to_d_1 %c to %d : exec_unit, memory
    datapath @c_to_d_2 %c to %d : exec_unit, memory
    yield %c
  } -> exec_unit

  datapath @b_to_c %b to %c : exec_unit, exec_unit
  datapath @c_to_b %c to %b : exec_unit, exec_unit
}