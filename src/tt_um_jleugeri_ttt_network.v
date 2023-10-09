
module tt_um_jleugeri_ttt_network #(
    int NEW_TOKENS_BITS = 4,
    int NUM_PROCESSORS = 10
) (
    // event start/stop inputs
    input logic [1:0] tstartstop[NUM_PROCESSORS],
    // good/bad event count outputs
    output logic [NEW_TOKENS_BITS-1:0] new_good_tokens[NUM_PROCESSORS],
    output logic [NEW_TOKENS_BITS-1:0] new_bad_tokens[NUM_PROCESSORS],
    // control signal
    output logic enable
);

    // internal control wires
    logic mux_hot;
    logic mux_go;
    logic mux_done;
    logic mux_next;

    // internal address wires
    logic [$log2(NUM_PROCESSORS)-1:0] idx_src;
    logic [(3*$log2(NUM_PROCESSORS))-1:0] tgt_range;
    logic [$log2(NUM_PROCESSORS)-1:0] idx_tgt;

    // internal data wires
    logic src_tstart;
    logic src_tstop;
    logic signed [NEW_TOKENS_BITS-1:0] tgt_new_good_tokens;
    logic signed [NEW_TOKENS_BITS-1:0] tgt_new_bad_tokens;

    // internal helper variables
    logic signed src_sign = src_tstart - src_tstop;
    logic [$log2(NUM_PROCESSORS)-1:0] mux_counter;
    logic [(2*$log2(NUM_PROCESSORS))-1:0] tgt_counter;

endmodule