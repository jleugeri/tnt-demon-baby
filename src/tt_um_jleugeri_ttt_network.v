


module tt_um_jleugeri_ttt_mux #(
    parameter int NUM_PROCESSORS
) (
    // control I/Os
    output logic hot_out,
    output logic done_out,
    input logic go_in,
    input logic next_in,
    // mux inputs
    input logic [2*NUM_PROCESSORS-1:0] t_startstop_in,
    // mux outputs
    output logic [$clog2(NUM_PROCESSORS)-1:0] idx_out,
    output logic t_start_out,
    output logic t_stop_out
);

    genvar gen_i;
    // generate
    //     for (gen_i=0; gen_i<NUM_PROCESSORS; gen_i=gen_i+1) begin : mux_loop
    //         logic 
    //     end
    // endgenerate
    
    // the MUX iterates over all processors with 

endmodule

module tt_um_jleugeri_ttt_network #(
    parameter int NEW_TOKENS_BITS = 4,
    parameter int NUM_PROCESSORS = 10
) (
    // event start/stop inputs
    input logic [2*NUM_PROCESSORS-1:0] tstartstop,
    // good/bad event count outputs
    output logic [NUM_PROCESSORS*NEW_TOKENS_BITS-1:0] new_good_tokens,
    output logic [NUM_PROCESSORS*NEW_TOKENS_BITS-1:0] new_bad_tokens,
    // control signal
    output logic enable
);

    // internal control wires
    logic mux_hot;
    logic mux_go;
    logic mux_done;
    logic mux_next;

    // internal address wires
    logic [$clog2(NUM_PROCESSORS)-1:0] idx_src;
    logic [(3*$clog2(NUM_PROCESSORS))-1:0] tgt_range;
    logic [$clog2(NUM_PROCESSORS)-1:0] idx_tgt;

    // internal data wires
    logic src_tstart;
    logic src_tstop;
    logic signed [NEW_TOKENS_BITS-1:0] tgt_new_good_tokens;
    logic signed [NEW_TOKENS_BITS-1:0] tgt_new_bad_tokens;

    // internal helper variables
    logic signed src_sign = src_tstart - src_tstop;
    logic [$clog2(NUM_PROCESSORS)-1:0] mux_counter;
    logic [(2*$clog2(NUM_PROCESSORS))-1:0] tgt_counter;

    // implement the multiplexer


    // implement the demultiplexer

endmodule