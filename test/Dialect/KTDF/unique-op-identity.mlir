// RUN: dataflow-scheduler-dialects-opt --cse %s | FileCheck %s

// CHECK-LABEL: func @cse_on_create_token
func.func @cse_on_create_token() -> (!ktdf.token, !ktdf.token) {
    // CHECK: %[[TOKA:.+]] = ktdf.create_token
    %a = ktdf.create_token : !ktdf.token
    // CHECK: %[[TOKB:.+]] = ktdf.create_token
    %b = ktdf.create_token : !ktdf.token
    // CHECK: return %[[TOKA]], %[[TOKB]]
    return %a, %b : !ktdf.token, !ktdf.token
}

// CHECK-LABEL: func @cse_on_tiling_reserve_size
func.func @cse_on_tiling_reserve_size() -> (index, index) {
    // CHECK: %[[SZA:.+]] = ktdf.tiling.reserve_size
    %a = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    // CHECK: %[[SZB:.+]] = ktdf.tiling.reserve_size
    %b = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    // CHECK: return %[[SZA]], %[[SZB]]
    return %a, %b : index, index
}
