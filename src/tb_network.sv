module tb_network();

    localparam NUM_PROCESSORS = 10;
    localparam NUM_CONNECTIONS = 50;
    localparam NEW_TOKEN_BITS = 4;

    // control inputs / outputs
    logic clk;
    logic reset;
    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id;
    logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id;
    logic done;
    logic valid;
    
    // outputs to processor
    logic [$clog2(NUM_PROCESSORS)-1:0] target_id;
    logic signed [NEW_TOKEN_BITS-1:0] new_good_tokens;
    logic signed [NEW_TOKEN_BITS-1:0] new_bad_tokens;

    // programming
    logic [2:0] instruction;
    logic [NEW_TOKEN_BITS-1:0] prog_tokens;

    initial begin
        clk = 0;
        reset = 0;
        processor_id = 0;
        connection_id = 0;
        instruction = 0;
        prog_tokens = 0;
    end

    // instantiate the connections
    tt_um_jleugeri_ttt_network #(
        .NUM_PROCESSORS(10),
        .NUM_CONNECTIONS(50),
        .NEW_TOKEN_BITS(4) 
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
        .prog_tokens(prog_tokens)
    );

endmodule: tb_network
