module tb_main();

    localparam int NUM_PROCESSORS = 10;
    localparam int NUM_CONNECTIONS = 50;
    localparam int NEW_TOKENS_BITS = 4;
    localparam int TOKENS_BITS = 8;
    localparam int DATA_BITS = 8;
    localparam int PROG_WIDTH = 8;
    localparam int DURATION_BITS = 8;

    // control flow logic
    logic reset;
    logic clock_fast;
    logic clock_slow;
    logic [2:0] stage;
    // data I/O logic
    logic signed [NEW_TOKENS_BITS-1:0] good_tokens_in, bad_tokens_in;
    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id,  processor_id_out;
    logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id;
    logic [1:0] token_startstop;
    logic output_valid;
    // programming logic
    logic [4:0] instruction;
    logic [PROG_WIDTH-1:0] prog_data;


    // instantiate the main module
    tt_um_jleugeri_ttt_main #(
        .NUM_PROCESSORS(NUM_PROCESSORS),
        .NUM_CONNECTIONS(NUM_CONNECTIONS),
        .NEW_TOKENS_BITS(NEW_TOKENS_BITS),
        .TOKENS_BITS(TOKENS_BITS),
        .DATA_BITS(DATA_BITS),
        .PROG_WIDTH(PROG_WIDTH),
        .DURATION_BITS(DURATION_BITS)
    ) main (
        // control flow logic
        .reset(reset),
        .clock_fast(clock_fast),
        .clock_slow(clock_slow),
        .stage(stage),
        // data I/O logic
        .good_tokens_in(good_tokens_in),
        .bad_tokens_in(bad_tokens_in),
        .processor_id_in(processor_id),
        .processor_id_out(processor_id_out),
        .token_startstop(token_startstop),
        .output_valid(output_valid),
        // programming logic
        .instruction(instruction),
        .prog_data(prog_data),
        .connection_id_in(connection_id)
    );

endmodule: tb_main
