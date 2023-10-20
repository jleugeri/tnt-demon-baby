`default_nettype none

module tt_um_jleugeri_ticktocktokens (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Bidirectional Input path
    output wire [7:0] uio_out,  // IOs: Bidirectional Output path
    output wire [7:0] uio_oe,   // IOs: Bidirectional Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // will go high when the design is enabled
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

    localparam int NUM_PROCESSORS = 10;
    localparam int NUM_CONNECTIONS = 50;
    localparam int NEW_TOKENS_BITS = 4;
    localparam int TOKENS_BITS = 8;
    localparam int DATA_BITS = 8;
    localparam int PROG_HEADER = 4;
    localparam int PROG_WIDTH = 8;
    localparam int DURATION_BITS = 8;

    // get positive reset
    wire reset = !rst_n;

    logic clock_fast;
    logic clock_slow;
    logic hold;
    logic [TOKENS_BITS-1:0] tokens_in;
    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id;
    logic [PROG_HEADER-1:0] prog_header;
    logic [PROG_WIDTH-1:0] prog_data;
    logic [1:0] token_startstop;

    logic done;
    logic [2:0] stage;

    // set up direction of bidirectional IOs
    assign uio_oe = 8'b00000000;
    assign uo_out[7:6] = 2'b11;
    assign uio_out = 8'b11111111;

    assign clock_fast = clk;
    assign clock_slow = 1;
    assign hold = 1;
    assign tokens_in = 8'b00000000;
    assign processor_id = 4'b0000;
    
    assign uo_out[1:0] = token_startstop;
    assign prog_header = uio_in[7:4];
    assign prog_data = ui_in;

    // instantiate the design
    tt_um_jleugeri_ticktocktokens_main #(
        .NUM_PROCESSORS(NUM_PROCESSORS),
        .NUM_CONNECTIONS(NUM_CONNECTIONS),
        .NEW_TOKENS_BITS(NEW_TOKENS_BITS),
        .TOKENS_BITS(TOKENS_BITS),
        .DATA_BITS(DATA_BITS),
        .PROG_HEADER(PROG_HEADER),
        .PROG_WIDTH(PROG_WIDTH),
        .DURATION_BITS(DURATION_BITS)
    ) main (
        // control flow logic
        .reset(reset),
        .clock_fast(clock_fast),
        .clock_slow(clock_slow),
        .hold(hold),
        .done(done),
        .stage(stage),
        // data I/O logic
        .tokens_in(tokens_in),
        .processor_id(processor_id),
        .token_startstop(token_startstop),
        // programming logic
        .prog_header(prog_header),
        .prog_data(prog_data)
    );

endmodule : tt_um_jleugeri_ticktocktokens