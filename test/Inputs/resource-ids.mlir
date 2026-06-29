ktdf_arch.device @device attributes {version = 1} {
  exec_unit {expected_id = "exec_unit"}
  group {expected_id = "group"} share() {
    exec_unit {expected_id = "exec_unit_2"}
    exec_unit @exec_unit_1 {expected_id = "exec_unit_1"}
  }
  exec_unit {kind = "special", expected_id = "special"}
}
