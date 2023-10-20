module tb_network();

    localparam NUM_PROCESSORS = 10;
    localparam NUM_CONNECTIONS = 50;
    localparam NEW_TOKENS_BITS = 4;
    localparam PROG_WIDTH = 8;

    // control inputs / outputs
    logic clk;
    logic reset;
    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id;
    logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id;
    logic done;
    logic valid;
    
    // outputs to processor
    logic [$clog2(NUM_PROCESSORS)-1:0] target_id;
    logic signed [NEW_TOKENS_BITS-1:0] new_good_tokens;
    logic signed [NEW_TOKENS_BITS-1:0] new_bad_tokens;

    // programming
    logic [2:0] instruction;
    logic [PROG_WIDTH-1:0] prog_data;

    // instantiate the connections
    tt_um_jleugeri_ttt_network #(
        .NUM_PROCESSORS(10),
        .NUM_CONNECTIONS(50),
        .NEW_TOKENS_BITS(4) 
    ) net (
        // control inputs / outputs
        .clk(clk),
        .reset(reset),
        .processor_id(processor_id),
        .connection_id(connection_id),
        .done(done),
        .valid(valid),
        
        // outputs to processor
        .target_id(target_id),
        .new_good_tokens(new_good_tokens),
        .new_bad_tokens(new_bad_tokens),

        // programming inputs
        .instruction(instruction),
        .prog_data(prog_data)
    );

endmodule: tb_network
