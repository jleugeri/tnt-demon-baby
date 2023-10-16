/*
this testbench just instantiates the module and makes some convenient wires
that can be driven / tested by the cocotb test.py
*/

// testbench is controlled by test.py
module tb_processor_core ();

    // this part dumps the trace to a vcd file that can be viewed with GTKWave

    localparam NEW_TOKENS_BITS = 8;
    localparam TOKENS_BITS = 8;
    localparam DURATION_BITS = 8;
    localparam NUM_PROCESSORS = 10;
    localparam PROG_WIDTH = 8;

    // wire up the inputs and outputs
    logic clock_fast;
    logic clock_slow;
    logic reset;
    logic hold;
    logic [$clog2(NUM_PROCESSORS)-1:0] neuron_id;
    logic [NEW_TOKENS_BITS-1:0] new_good_tokens;
    logic [NEW_TOKENS_BITS-1:0] new_bad_tokens;
    logic token_start;
    logic token_stop;
    logic [2:0] prog_header;
    logic [PROG_WIDTH-1:0] prog_data;

    // instantiate just the processor core by itself
    tt_um_jleugeri_ttt_processor_core #(
        .NEW_TOKENS_BITS(NEW_TOKENS_BITS),
        .TOKENS_BITS(TOKENS_BITS),
        .DURATION_BITS(DURATION_BITS),
        .NUM_PROCESSORS(NUM_PROCESSORS),
        .PROG_WIDTH(PROG_WIDTH)
    ) proc (
        // control inputs
        .clock_fast(clock_fast),
        .clock_slow(clock_slow),
        .reset(reset),
        .hold(hold),
        .neuron_id(neuron_id),
        // data inputs
        .new_good_tokens(new_good_tokens),
        .new_bad_tokens(new_bad_tokens),
        // data outputs
        .token_start(token_start),
        .token_stop(token_stop),
        // programming inputs
        .prog_header(prog_header),
        .prog_data(prog_data)
    );

endmodule : tb_processor_core
