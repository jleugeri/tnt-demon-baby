
module tt_um_jleugeri_ttt_main #(
    parameter int NUM_PROCESSORS = 10,
    parameter int NUM_CONNECTIONS = 50,
    parameter int NEW_TOKENS_BITS = 4,
    parameter int TOKENS_BITS = 8,
    parameter int DATA_BITS = 8,
    parameter int PROG_WIDTH = 8,
    parameter int DURATION_BITS = 8
) (
    // control flow logic
    input  logic reset,
    input  logic clock_fast,
    input  logic clock_slow,
    output logic [2:0] stage,
    // data I/O logic
    input  logic [$clog2(NUM_PROCESSORS)-1:0] processor_id_in,
    output logic [$clog2(NUM_PROCESSORS)-1:0] processor_id_out,
    input  logic [NEW_TOKENS_BITS-1:0] good_tokens_in,
    input  logic [NEW_TOKENS_BITS-1:0] bad_tokens_in,
    output logic [1:0] token_startstop,
    output logic output_valid,
    // programming logic
    input logic [4:0] instruction,
    input logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id_in,
    input logic [PROG_WIDTH-1:0] prog_data
);
    localparam logic[$clog2(NUM_PROCESSORS+1)-1:0] MAX_PROCESSOR_ID_PLUS_ONE = ($clog2(NUM_PROCESSORS+1))'(NUM_PROCESSORS);
    localparam logic[$clog2(NUM_PROCESSORS+1)-1:0] MAX_PROCESSOR_ID = ($clog2(NUM_PROCESSORS+1))'(NUM_PROCESSORS-1);

    // internal control wires
    logic network_done;
    logic network_valid;
    logic [2:0] instruction_net;
    logic [2:0]  instruction_proc;
    logic [PROG_WIDTH-1:0] proc_prog_data;
    logic [PROG_WIDTH-1:0] net_prog_data;
    
    // internal address wires
    logic [$clog2(NUM_PROCESSORS)-1:0] source_id;
    logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id;
    logic [$clog2(NUM_PROCESSORS)-1:0] target_id;

    // internal data wires
    logic [1:0] token_startstop_internal;
    logic signed [NEW_TOKENS_BITS-1:0] tgt_new_good_tokens;
    logic signed [NEW_TOKENS_BITS-1:0] tgt_new_bad_tokens;
    logic signed [NEW_TOKENS_BITS-1:0] net_new_good_tokens, net_new_bad_tokens, proc_new_good_tokens, proc_new_bad_tokens;

    // instantiate the start/stop token buffer
    logic token_start_buf [NUM_PROCESSORS-1:0];
    logic token_stop_buf [NUM_PROCESSORS-1:0];

    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id;
    logic [$clog2(NUM_PROCESSORS+1)-1:0] processor_id_internal, processor_id_internal_prev;

    // instantiate the network
    tt_um_jleugeri_ttt_network #(
        .NUM_PROCESSORS(10),
        .NUM_CONNECTIONS(50),
        .NEW_TOKENS_BITS(4) 
    ) net (
        // control inputs / outputs
        .clk(clock_fast),
        .reset(reset),
        .processor_id(source_id),
        .connection_id(connection_id),
        .done(network_done),
        .valid(network_valid),
        
        // outputs to processor
        .target_id(target_id),
        .new_good_tokens(net_new_good_tokens),
        .new_bad_tokens(net_new_bad_tokens),

        // programming inputs
        .instruction(instruction_net),
        .prog_data(net_prog_data)
    );


    // instantiate the processor core
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
        .processor_id(processor_id),
        // data inputs
        .new_good_tokens(proc_new_good_tokens),
        .new_bad_tokens(proc_new_bad_tokens),
        // data outputs
        .token_startstop(token_startstop_internal),
        // programming inputs
        .instruction(instruction_proc),
        .prog_data(proc_prog_data)
    );

    logic first_cycle;

    always_ff @( posedge clock_fast ) begin
        if (reset) begin
            // next stage should be INPUT
            stage <= 3'b000;
        end
        else begin
            // if the MSB of instruction is high, we are in the programming mode
            // the second MSB in the programming mode determines whether we're configuring the network or the processors
            case (instruction[4:3])
                2'b11 : begin
                    // if the second MSB is set, configure the network
                    instruction_net <= instruction[2:0];
                    source_id <= processor_id_in;
                    connection_id <= connection_id_in;
                    net_prog_data <= prog_data;
                end
                // alternatively, configure the processor
                2'b10 : begin
                    instruction_proc <= instruction[2:0];
                    processor_id <= processor_id_in;
                    proc_prog_data <= prog_data;
                end
                // else, we are in execution mode
                default: begin
                    // cycle between stages:
                    // 1. while there is external input pending or advancement has been halted (input hold is asserted)
                    //    - read that external input
                    //    - update the corresponding processor's input (stage 1)
                    //    - repeat until hold is deasserted
                    // 2. iterate over all processors, and for each:
                    //    - update its internal state (stage2)
                    //    - read out any generated tokens (lagging behind one cycle!) and store them in the new token buffer
                    // 3. loop over the non-zero elements in the new token buffer, and for each:
                    //    - loop over all outgoing connections, and for each:
                    //          - take the generated token output (lagging behind one cycle!)
                    //          - update the corresponding processor's input (stage 1)
                    case (stage)

                        // stage 1: receive external inputs
                        3'b000 : begin
                            // the three LSB of the instruction select the operation
                            case (instruction[2:0])

                                // read input
                                3'b001: begin
                                    instruction_proc <= 3'b100;
                                    processor_id <= processor_id_in;
                                    proc_new_good_tokens <= good_tokens_in;
                                    proc_new_bad_tokens <= bad_tokens_in;
                                end

                                // advance to the next stage
                                3'b010: begin
                                    stage <= 3'b001;
                                    // set up for looping over all processors
                                    processor_id <= 0;
                                    processor_id_internal <= 0;
                                    processor_id_internal_prev <= -1;
                                    proc_new_good_tokens <= 0;
                                    proc_new_bad_tokens <= 0;
                                    instruction_proc <= 3'b101;
                                end

                                default: begin
                                    // block!
                                end
                            endcase
                        end

                        // stage 2: update all processors
                        3'b001 : begin

                            // outputs of the processors arrive with a one-cycle delay
                            // if this was not the first clock cycle, store the output of the previous processor
                            if(processor_id_internal != 0) begin
                                token_start_buf[processor_id_internal_prev] <= token_startstop_internal[1];
                                token_stop_buf[processor_id_internal_prev] <= token_startstop_internal[0];
                            end

                            //advance to next stage
                            if (processor_id_internal == MAX_PROCESSOR_ID_PLUS_ONE) begin
                                stage <= 3'b010;
                                processor_id_internal <= 0;
                            end
                            // in the penultimate step, don't update the processors any more; 
                            else if (processor_id_internal == MAX_PROCESSOR_ID) begin
                                instruction_proc <= 3'b000;
                                processor_id_internal_prev <= processor_id_internal;
                                processor_id_internal <= processor_id_internal + 1;
                                processor_id <= 0;
                            end
                            else begin
                                // update the next processor and advance
                                processor_id_internal_prev <= processor_id_internal;
                                processor_id_internal <= processor_id_internal + 1;
                                processor_id <= processor_id_internal + 1;
                            end
                        end

                        // stage 3
                        // go through all processors, check if they fired a token
                        3'b010 : begin
                            if (processor_id_internal == MAX_PROCESSOR_ID_PLUS_ONE) begin
                                // if we reached the end, go back to waiting for external input
                                stage <= 3'b000;
                            end
                            else begin
                                // cycle over all processors, and for each, cycle over all connections before moving on to the next processor
                                if (token_start_buf[processor_id_internal] ^ token_stop_buf[processor_id_internal]) begin
                                    // write output back to host
                                    processor_id_out <= processor_id_internal;
                                    token_startstop <= {token_start_buf[processor_id_internal], token_stop_buf[processor_id_internal]};
                                    output_valid <= 1;
                                    // if the processor started/stopped a token, start cycling over its connections

                                    stage <= 3'b011;
                                    instruction_net <= 3'b110;
                                    instruction_proc <= 3'b000;
                                    source_id <= processor_id_internal;
                                    first_cycle <= 1;
                                end
                                else begin
                                    output_valid <= 0;
                                end
                                // advance to the next processor
                                processor_id_internal_prev <= processor_id_internal;
                                processor_id_internal <= processor_id_internal + 1;
                            end
                        end

                        3'b011 : begin
                            // continue cycling
                            first_cycle <= 0;

                            token_startstop <= 2'b00;
                            output_valid <= 0;

                            // if we're done iterating, go to the next processor
                            if (network_done) begin
                                instruction_net <= 3'b000;
                                stage <= 3'b010;
                                instruction_proc <= 3'b000;
                            end 
                            else begin
                                instruction_net <= 3'b111;
                                if (!first_cycle) begin
                                    // if the processor started a token, use positive weights, otherwise, negative
                                    
                                    if (token_start_buf[processor_id_internal_prev]) begin
                                        proc_new_good_tokens <= net_new_good_tokens;
                                        proc_new_bad_tokens <= net_new_bad_tokens;
                                    end
                                    else begin
                                        proc_new_good_tokens <= -net_new_good_tokens;
                                        proc_new_bad_tokens <= -net_new_bad_tokens;
                                    end

                                    if (network_valid) begin
                                        processor_id <= target_id;
                                        instruction_proc <= 3'b100;
                                    end
                                end
                            end
                        end

                        // otherwise do nothing
                        default: begin
                        end
                    endcase
                end
            endcase
        end
    end

/*
STUFF from old mux object:

    // external inputs
    input logic has_ext_input,
    input logic [$clog2(NUM_PROCESSORS)-1:0] ext_tgt_addr,
    input logic signed [NEW_TOKENS_BITS-1:0] new_ext_good_tokens,
    input logic signed [NEW_TOKENS_BITS-1:0] new_ext_bad_tokens,


    // cache of the processors' inputs
    logic [NEW_TOKENS_BITS-1:0] new_good_tokens_cache[NUM_PROCESSORS-1:0];
    logic [NEW_TOKENS_BITS-1:0] new_bad_tokens_cache[NUM_PROCESSORS-1:0];


                
                // when cycling connections, each cycle we update one processor and then start cycling its connections if it spiked
                // if on spike is generated, we directly move on to the next processor
                CYCLE_PROCESSORS: begin
                    // process the current processor's input
                    new_good_tokens <= new_good_tokens_cache[i];
                    new_bad_tokens <= new_bad_tokens_cache[i];

                    // process the last step's output
                    if (token_start || token_stop) begin
                        // read the current processor's output and process it
                        next_state <= CYCLE_CONNECTIONS;
                        cycle_complete <= 0;
                        // get start and end addresses for the current processor's connections
                        j <= tgt_addr_first[i];
                        end_addr <= tgt_addr_first[i+1]-1;
                    end else begin 
                        // keep going until we took care of all processors
                        if (i == NUM_PROCESSORS-1) begin
                            // if we reached the end, wrap around ...
                            i <= 0;
                            cycle_complete <= 1;
                            if (has_ext_input) begin
                                // ... and start cycling through external inputs (if desired), ...
                                next_state <= CYCLE_EXTERNAL_INPUT;
                            end else begin
                                // ... otherwise, continue with the next processor
                                next_state <= CYCLE_PROCESSORS;
                            end
                        end else begin
                            // increment the counter and move on to the next processor
                            next_state <= CYCLE_PROCESSORS;
                            cycle_complete <= 0;
                            i <= i + 1;
                        end
                    end
                end

                // when cycling external inputs, we take one external input each cycle and add it to the correct buffer
                CYCLE_EXTERNAL_INPUT: begin
                    // add the external input to the correct buffer
                    new_good_tokens_cache[ext_tgt_addr] <= new_good_tokens_cache[ext_tgt_addr] + new_ext_good_tokens;
                    new_bad_tokens_cache[ext_tgt_addr] <= new_bad_tokens_cache[ext_tgt_addr] + new_ext_bad_tokens;

                    // if we have more external inputs, stay in this state
                    // otherwise, start the next cycle over the processors
                    if (has_ext_input) begin
                        next_state <= CYCLE_EXTERNAL_INPUT;
                        cycle_complete <= 0;
                    end else begin
                        next_state <= CYCLE_PROCESSORS;
                        cycle_complete <= 1;
                        i <= 0;
                    end
                end
*/
endmodule : tt_um_jleugeri_ttt_main
