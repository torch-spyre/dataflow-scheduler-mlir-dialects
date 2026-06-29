// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt --allow-unregistered-dialect

#core = { kind = "core" }
#corelet = { kind = "corelet" }
#corelet_private = { kind = "corelet_private" }

ktdf_arch.device @ring_corelet_device {
  %core_shared = memory { kind = "core_shared" }
  
  group share(%core_shared) {
    %cross02 = memory { kind = "cross_shared" }
    %cross13 = memory { kind = "cross_shared" }

    %corelet0, %corelet1 = group @core0 #core share(%core_shared, %cross02, %cross13) {
      %corelet_shared = memory { kind = "corelet_shared" }
      datapath %core_shared to %corelet_shared : memory, memory
      datapath %corelet_shared to %core_shared : memory, memory

      %corelet0 = group @corelet0 #corelet share(%corelet_shared, %cross02) {
        %corelet_private = memory #corelet_private
        datapath %corelet_shared to %corelet_private : memory, memory
        datapath %corelet_private to %corelet_shared : memory, memory
        datapath %cross02 to %corelet_private : memory, memory
        datapath %corelet_private to %cross02 : memory, memory

        %corelet = exec_unit
        datapath %corelet_private to %corelet : memory, exec_unit
        datapath %corelet to %corelet_private : exec_unit, memory
        yield %corelet
      } -> exec_unit

      %corelet1 = group @corelet1 #corelet share (%corelet_shared, %cross13) {
        %corelet_private = memory #corelet_private
        datapath %corelet_shared to %corelet_private : memory, memory
        datapath %corelet_private to %corelet_shared : memory, memory 
        datapath %cross13 to %corelet_private : memory, memory
        datapath %corelet_private to %cross13 : memory, memory

        %corelet = exec_unit
        datapath %corelet_private to %corelet : memory, exec_unit
        datapath %corelet to %corelet_private : exec_unit, memory
        yield %corelet
      } -> exec_unit

      datapath %corelet0 to %corelet1 : exec_unit, exec_unit
      yield %corelet0, %corelet1
    } -> exec_unit, exec_unit

    %corelet2, %corelet3 = group @core1 #core share(%core_shared, %cross02, %cross13) {
      %corelet_shared = memory { kind = "corelet_shared" }
      datapath %core_shared to %corelet_shared : memory, memory
      datapath %corelet_shared to %core_shared : memory, memory

      %corelet2 = group @corelet2 #corelet share(%corelet_shared, %cross02) {
        %corelet_private = memory #corelet_private
        datapath %corelet_shared to %corelet_private : memory, memory
        datapath %corelet_private to %corelet_shared : memory, memory
        datapath %cross02 to %corelet_private : memory, memory
        datapath %corelet_private to %cross02 : memory, memory

        %corelet = exec_unit
        datapath %corelet_private to %corelet : memory, exec_unit
        datapath %corelet to %corelet_private : exec_unit, memory
        yield %corelet
      } -> exec_unit
      
      %corelet3 = group @corelet3 #corelet share(%corelet_shared, %cross13) {
        %corelet_private = memory #corelet_private
        datapath %corelet_shared to %corelet_private : memory, memory
        datapath %corelet_private to %corelet_shared : memory, memory
        datapath %cross13 to %corelet_private : memory, memory
        datapath %corelet_private to %cross13 : memory, memory

        %corelet = exec_unit
        datapath %corelet_private to %corelet : memory, exec_unit
        datapath %corelet to %corelet_private : exec_unit, memory
        yield %corelet
      } -> exec_unit

      datapath %corelet2 to %corelet3 : exec_unit, exec_unit
      yield %corelet2, %corelet3
    } -> exec_unit, exec_unit

    datapath %corelet1 to %corelet2 : exec_unit, exec_unit
    datapath %corelet3 to %corelet0 : exec_unit, exec_unit
  }
}
