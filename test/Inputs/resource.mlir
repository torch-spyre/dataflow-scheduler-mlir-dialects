ktdf_arch.device @my_device attributes {version = 1} {
  %memory = memory @memory { kind = "space" }

  %exec_unit = exec_unit @exec_unit { 
    ktdf_arch.features = { 
      ktdf_arch.feature.compute, 
      ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f32 = 4, f16 = 8> } 
    }
  }

  datapath @datapath { 
    ktdf_arch.bandwidth = 128, 
    ktdf_arch.features = { 
      ktdf_arch.feature.queue = { size = 4096, depth = 256, ordered } 
    }  
  } %memory to %exec_unit : memory, exec_unit
}