// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics -allow-unregistered-dialect %s

func.func @private_with_invalid_child() {
    // expected-error@+1 {{immediate children must be 'ktdf.stage' or 'ktdf.private' ops}}
    ktdf.pipeline {
        // expected-note@+1 {{found 'unregistered.op'}}
        "unregistered.op"() : () -> ()
    }
    return
}

// -----

func.func @private_with_multiple_private() {
    // expected-error@+1 {{at most one immediate 'ktdf.private' child is allowed}}
    ktdf.pipeline {
        // expected-note@+1 {{previous 'ktdf.private' is here}}
        ktdf.private {}
        ktdf.private {}
    }
    return
}