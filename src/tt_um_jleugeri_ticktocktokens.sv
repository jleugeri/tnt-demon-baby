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
    // Instructions:
    //  - XX--: we need to distinguish programming the processor, programming the network, and running the system (2 bit)
    //  - 00XX: for execution mode, we need to distinguish between different commands (2 bit)
    //    - 0000: don't advance to next stage (no-op)
    //    - 0001: read an input signal (data: log2(NUM_PROCESSORS) + (2 or 1)x NEW_TOKEN_BITS bit))
    //    - 0010: advance to the next stage
    //    - 0011: (reserved)
    //    - 01XX: (reserved)
    //  - 1XXX: for programming mode, we need to distinguish between programming processor and network (1 bit)
    //    - 10XX: for programming the processor, we need a 2 bit operator + data bits:
    //      - 1000: (reserved)
    //      - 1001: program duration  (data: log2(NUM_PROCESSORS) + DURATION_BITS bit)
    //      - 1010: program good token threshold log2(NUM_PROCESSORS) + (data: TOKEN_BITS bit)
    //      - 1011: programd bad token threshold log2(NUM_PROCESSORS) + (data: TOKEN_BITS bit)
    //    - 11XX: for programming the network, we need a 3 bit operator + data bits:
    //      - 1100: good weights (data: log2(NUM_CONNECTIONS) + NEW_TOKEN_BITS bit)
    //      - 1101: bad weights (data: log2(NUM_CONNECTIONS) + NEW_TOKEN_BITS bit)
    //      - 1110: indptr (data: log2(NUM_PROCESSORS) + log2(NUM_CONNECTIONS) bit)
    //      - 1111: indices (data: log2(NUM_CONNECTIONS) + log2(NUM_PROCESSORS) bit)
    //
    // Assuming a system with:
    //  - 16 processors => log2(NUM_PROCESSORS) = 4
    //  - up to 256 connections => log2(NUM_CONNECTIONS) = 8
    //  - up to 16 good and up to 16 bad tokens per weight => NEW_TOKEN_BITS = 4
    //  - a total of up to 256 tokens => TOKEN_BITS = 8
    //  - up to 256 cycles token duration => DURATION_BITS = 8
    // we need the following amount of data for each instruction (in addition to the instruction itself):
    //  - 0001: 4 + 2*4 = 12 bits if we write both good and bad tokens simulataneously
    //          4 + 4 = 8 bits if we write only good or bad tokens, respectively, in one cycle
    //  - 1001: 4 + 8 = 12 bits
    //  - 1010: 4 + 8 = 12 bits
    //  - 1011: 4 + 8 = 12 bits
    //  - 1100: 8 + 4 = 12 bits
    //  - 1101: 8 + 4 = 12 bits
    //  - 1110: 4 + 8 = 12 bits
    //  - 1111: 8 + 4 = 12 bits
    //
    // So every instruction needs 16 bit, 4 bit for the instruction and 12 bit for the data.
    // We set all IO pins to input to accomodate this.
    //
    // The package format is thus:
    //
    // |<--    ui_in (8bit)  -->|<--    uio_in (8bit) -->|
    // +------------+------------------------------------+
    // |  op (4bit) |            data  (12bit)           |
    // +------------+------------------------------------+
    //
    // where for each instruction, the data is interpreted as follows:
    //    0001:     |proc (4bit)| good (4bit)| bad (4bit)| 
    //
    //    1001:     |proc (4bit)|    duration (8bit)     |
    //    1010:     |proc (4bit)|   good thresh (8bit)   |
    //    1011:     |proc (4bit)|    bad thresh (8bit)   |
    //    1100:     |good (4bit)|  connection ID (8bit)  |
    //    1101:     |bad (4bit) |  connection ID (8bit)  |
    //    1110:     |proc (4bit)|  connection ID (8bit)  |
    //    1111:     |proc (4bit)|  connection ID (8bit)  |
    //
    // For the output, we have the following format:
    //
    // +------------+-----------+
    // |  proc ID  |st/sp | stat | 
    // |  (4bit)   |(2bit)|(2bit)|
    // +------------+-----------+
    // where:
    //  - proc ID: the ID of the processor that just fired (if any)
    //  - st/sp: start/stop token for the processor:
    //    - 00: no token
    //    - 01: start token
    //    - 10: stop token
    //    - 11: start and stop token
    //  - stat: status of the execution:
    //    - 00: waiting for external signals
    //    - 01: updating processors' internal states
    //    - 10: checking processors for new tokens
    //    - 11: transmitting tokens via network



    localparam int NUM_PROCESSORS = 4;
    localparam int NUM_CONNECTIONS = 12;
    localparam int NEW_TOKEN_BITS = 4;
    localparam int TOKEN_BITS = 8;
    localparam int DURATION_BITS = 8;

    // data I/O logic
    logic signed [NEW_TOKEN_BITS-1:0] good_tokens_in, bad_tokens_in;
    logic [$clog2(NUM_PROCESSORS+1)-1:0] processor_id;
    logic [$clog2(NUM_PROCESSORS)-1:0]  processor_id_out;
    logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id_in;
    logic [1:0] token_startstop;
    logic [1:0] stage;
    logic output_valid;

    // programming logic
    logic [3:0] instruction;
    logic [DURATION_BITS-1:0] prog_duration;
    logic [TOKEN_BITS-1:0] prog_threshold;
    logic [NEW_TOKEN_BITS-1:0] prog_tokens;


    // connect the wires

    // set all programmable IO pins to input
    assign uio_oe = 8'b00000000;

    // control flow logic
    logic reset;
    assign reset = ~rst_n;

    // for now, clock_slow = 1
    logic clock_slow;
    assign clock_slow = 1;

    // wire up inputs
    assign instruction = ui_in[7:4];

    // for execution mode
    assign processor_id = $clog2(NUM_PROCESSORS+1)'(ui_in[3:0]);
    assign good_tokens_in = NEW_TOKEN_BITS'(uio_in[7:4]);
    assign bad_tokens_in  = NEW_TOKEN_BITS'(uio_in[3:0]);

    // for programming mode
    assign prog_tokens = NEW_TOKEN_BITS'(ui_in[3:0]);
    assign connection_id_in = $clog2(NUM_CONNECTIONS)'(uio_in);
    assign prog_threshold = TOKEN_BITS'(uio_in);
    assign prog_duration = DURATION_BITS'(uio_in);

    // assign outputs
    assign uo_out = {4'(processor_id_out), token_startstop, stage};
    assign uio_out = 8'bZZZZZZZZ;

    // instantiate the main module
    tt_um_jleugeri_ttt_main #(
        .NUM_PROCESSORS(NUM_PROCESSORS),
        .NUM_CONNECTIONS(NUM_CONNECTIONS),
        .NEW_TOKEN_BITS(NEW_TOKEN_BITS),
        .TOKEN_BITS(TOKEN_BITS),
        .DURATION_BITS(DURATION_BITS)
    ) main (
        // control flow logic
        .reset(reset),
        .clock_fast(clk),
        .clock_slow(clock_slow),
        .instruction(instruction),
        .stage(stage),
        // data I/O logic
        .good_tokens_in(good_tokens_in),
        .bad_tokens_in(bad_tokens_in),
        .processor_id_in(processor_id),
        .processor_id_out(processor_id_out),
        .token_startstop(token_startstop),
        .output_valid(output_valid),
        // programming logic
        .connection_id_in(connection_id_in),
        .prog_tokens(prog_tokens),
        .prog_threshold(prog_threshold),
        .prog_duration(prog_duration)
    );


endmodule : tt_um_jleugeri_ticktocktokens
