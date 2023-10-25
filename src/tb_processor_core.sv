/*
this testbench just instantiates the module and makes some convenient wires
that can be driven / tested by the cocotb test.py
*/

// testbench is controlled by test.py
module tb_processor_core ();

    // this part dumps the trace to a vcd file that can be viewed with GTKWave

    localparam NEW_TOKEN_BITS = 8;
    localparam TOKEN_BITS = 8;
    localparam DURATION_BITS = 8;
    localparam NUM_PROCESSORS = 10;

    // wire up the inputs and outputs
    logic clock_fast;
    logic clock_slow;
    logic reset;
    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id;
    logic [NEW_TOKEN_BITS-1:0] new_good_tokens;
    logic [NEW_TOKEN_BITS-1:0] new_bad_tokens;
    logic [1:0] token_startstop;
    logic [2:0] instruction;
    logic [DURATION_BITS-1:0] prog_duration;
    logic [TOKEN_BITS-1:0] prog_threshold;

    initial begin
        clock_fast = 0;
        clock_slow = 0;
        reset = 0;
        processor_id = 0;
        new_good_tokens = 0;
        new_bad_tokens = 0;
        instruction = 0;
        prog_duration = 0;
        prog_threshold = 0;
    end

    // instantiate just the processor core by itself
    tt_um_jleugeri_ttt_processor_core #(
        .NEW_TOKEN_BITS(NEW_TOKEN_BITS),
        .TOKEN_BITS(TOKEN_BITS),
        .DURATION_BITS(DURATION_BITS),
        .NUM_PROCESSORS(NUM_PROCESSORS)
    ) proc (
        // control inputs
        .clock_fast(clock_fast),
        .clock_slow(clock_slow),
        .reset(reset),
        .processor_id(processor_id),
        // data inputs
        .new_good_tokens(new_good_tokens),
        .new_bad_tokens(new_bad_tokens),
        // data outputs
        .token_startstop(token_startstop),
        // programming inputs
        .instruction(instruction),
        .prog_duration(prog_duration),
        .prog_threshold(prog_threshold)
    );

endmodule : tb_processor_core
