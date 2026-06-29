ktdf_arch.device @inline_device {
  exec_unit
}

ktdf_arch.device @missing_device import("absolute-device.mlir")

ktdf_arch.device @device attributes {version = 2, overridable = 2} import("device.mlir")

ktdf_arch.device @invalid_device import("device-invalid.mlir")

ktdf_arch.device @recursive_device import("device-manager-test.mlir")
