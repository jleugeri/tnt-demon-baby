module tb_network();

    localparam NUM_PROCESSORS = 10;
    localparam NUM_CONNECTIONS = 50;
    localparam NEW_TOKENS_BITS = 4;

    // control inputs / outputs
    logic clk;
    logic reset;
    logic valid_in;
    logic [$clog2(NUM_PROCESSORS)-1:0] source_id;
    logic done;
    logic valid_out;

    // inputs from processor
    logic [1:0] token_startstop;
    
    // outputs to processor
    logic [$clog2(NUM_PROCESSORS)-1:0] target_id;
    logic signed [NEW_TOKENS_BITS-1:0] new_good_tokens;
    logic signed [NEW_TOKENS_BITS-1:0] new_bad_tokens;

    // instantiate the connections
    tt_um_jleugeri_ttt_network #(
        .NUM_PROCESSORS(10),
        .NUM_CONNECTIONS(50),
        .NEW_TOKENS_BITS(4) 
    ) net (
        // control inputs / outputs
        .clk(clk),
        .reset(reset),
        .valid_in(valid_in),
        .source_id(source_id),
        .done(done),
        .valid_out(valid_out),

        // inputs from processor
        .token_startstop(token_startstop),
        
        // outputs to processor
        .target_id(target_id),
        .new_good_tokens(new_good_tokens),
        .new_bad_tokens(new_bad_tokens)
    );

endmodule: tb_network
