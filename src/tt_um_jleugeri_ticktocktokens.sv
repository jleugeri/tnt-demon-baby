`default_nettype none

// Operations:
//  - Interact with the processor the normal way:
//    - 4'b0000: add/substract good tokens
//    - 4'b0001: add/subtract bad tokens
//  - Set/get the internal state directly:
//    - 4'b0010: set good token count
//    - 4'b0011: get good token count
//    - 4'b0100: set bad token count
//    - 4'b0101: get bad token count
//    - 4'b0110: set remaining duration (implicitly sets on-state)
//    - 4'b0111: get remaining duration (implicitly gets on-state)
//  - Special operations:
//    - 4'b1000: Tally up and make a decision (start/stop token?)
//    - 4'b1001: advance countdown (stop token?)
//  - Set/get the processor's parameters: 
//    - 4'b1010: set good token threshold
//    - 4'b1011: get good token threshold
//    - 4'b1100: set bad token threshold
//    - 4'b1101: get bad token threshold
//    - 4'b1110: set token duration
//    - 4'b1111: get token duration
//
// The input data format is thus:
//
// |<--   ui_in (8bit)  -->|<--   uio_in (8bit) -->|
// +-----------+-----------+-----------------------+
// | reserved  | op (4bit) |      data  (8bit)     |
// +-----------+-----------+-----------------------+
//
// where for each instruction, the values are packed into the LSBs of the data field.
//
// In response to every read/write instruction, the latest value is echoed back on uo_out
// (for read: the current value; for write: the just written value).
// 
// For the tally instruction (4'b1000), no input data is needed and the returned format is:
//
// |<--  uo_out (8bit)  -->|
// +-----------------+-----+
// |    reserved     |st|sp|
// +-----------------+-----+
//                     |  |
//                     |  `- token stop flag (1bit)
//                      `- token start flag (1bit) 

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

    localparam int TOKEN_BITS = 8;
    localparam int NEW_TOKEN_BITS = 8;
    localparam int DURATION_BITS = 8;
    localparam int DATA_BITS = 8;
    localparam int INSTRUCTION_BITS = 4;

    // unused pins
    wire [7:INSTRUCTION_BITS] unused_in;
    assign unused_in = ui_in[7:INSTRUCTION_BITS];

    // programming logic
    logic [INSTRUCTION_BITS-1:0] instruction;
    assign instruction = ui_in[INSTRUCTION_BITS-1:0];

    logic [DATA_BITS-1:0] data_from_host, data_from_proc;
    assign data_from_host = uio_in[DATA_BITS-1:0];

    // data I/O logic
    // breaking out the single "token input" signal into good and bad tokens, respectively, based on the MSB
    logic signed [NEW_TOKEN_BITS-1:0] good_tokens_in, bad_tokens_in;
    assign good_tokens_in = (instruction == 4'b0000) ? uio_in[NEW_TOKEN_BITS-1:0] : NEW_TOKEN_BITS'(1'b0);
    assign bad_tokens_in  = (instruction == 4'b0001) ? uio_in[NEW_TOKEN_BITS-1:0] : NEW_TOKEN_BITS'(1'b0);

    // combine the output signals
    logic token_start, token_stop, token_valid, expect_data;
    assign uo_out = expect_data ? data_from_proc : {5'b00000, token_valid, token_start, token_stop};

    // set all programmable IO pins to input
    assign uio_oe  = 8'b00000000;
    assign uio_out = 8'b00000000;

    // control flow logic
    logic reset;
    assign reset = ~rst_n;

    // instantiate the main module
    ttt_processor #(
        .NEW_TOKEN_BITS(NEW_TOKEN_BITS),
        .TOKEN_BITS(TOKEN_BITS),
        .DURATION_BITS(DURATION_BITS),
        .DATA_BITS(DATA_BITS),
        .INSTRUCTION_BITS(INSTRUCTION_BITS)
    ) proc (
        // control flow logic
        .reset(reset),
        .clock(clk),
        .enable(ena),
        .instruction(instruction),
        // data I/O logic
        .good_tokens_in(good_tokens_in),
        .bad_tokens_in(bad_tokens_in),
        .token_start(token_start),
        .token_stop(token_stop),
        .token_valid(token_valid),
        // programming logic
        .data_in(data_from_host),
        .data_out(data_from_proc)
    );

    // whenever we get instruction 4'b1000 or 4'b1001, we expect start/stop tokens back
    // otherwise, we expect data to be returned
    always_ff @( posedge clk ) begin
        if ( rst_n == 0 ) begin
            expect_data <= 1'b0;
        end else begin
            expect_data <= (instruction != 4'b1000);
        end
    end


endmodule : tt_um_jleugeri_ticktocktokens
