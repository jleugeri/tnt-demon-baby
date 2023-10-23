
module tt_um_jleugeri_ttt_main #(
    parameter int NUM_PROCESSORS = 10,
    parameter int NUM_CONNECTIONS = 50,
    parameter int NEW_TOKEN_BITS = 4,
    parameter int TOKEN_BITS = 8,
    parameter int DURATION_BITS = 8
) (
    // control flow logic
    input  logic reset,
    input  logic clock_fast,
    input  logic clock_slow,
    input logic [3:0] instruction,
    output logic [1:0] stage,
    // data I/O logic
    input  logic [$clog2(NUM_PROCESSORS+1)-1:0] processor_id_in,
    output logic [$clog2(NUM_PROCESSORS)-1:0] processor_id_out,
    input  logic signed [NEW_TOKEN_BITS-1:0] good_tokens_in,
    input  logic signed [NEW_TOKEN_BITS-1:0] bad_tokens_in,
    output logic [1:0] token_startstop,
    output logic output_valid,
    // programming logic
    input logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id_in,
    input logic [DURATION_BITS-1:0] prog_duration,
    input logic [TOKEN_BITS-1:0] prog_threshold,
    input logic [NEW_TOKEN_BITS-1:0] prog_tokens
);
    localparam logic [$clog2(NUM_PROCESSORS+1)-1:0] MAX_PROCESSOR_ID_PLUS_ONE = ($clog2(NUM_PROCESSORS+1))'(NUM_PROCESSORS);
    localparam logic [$clog2(NUM_PROCESSORS+1)-1:0] MAX_PROCESSOR_ID = ($clog2(NUM_PROCESSORS+1))'(NUM_PROCESSORS-1);

    // internal control wires
    logic network_done;
    logic network_valid;
    logic [2:0] instruction_net;
    logic [2:0]  instruction_proc;

    logic [DURATION_BITS-1:0] proc_prog_duration;
    logic [TOKEN_BITS-1:0] proc_prog_threshold;
    logic [NEW_TOKEN_BITS-1:0] net_prog_tokens;

    // internal address wires
    logic [$clog2(NUM_PROCESSORS+1)-1:0] source_id;
    logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id;
    logic [$clog2(NUM_PROCESSORS)-1:0] target_id;

    // internal data wires
    logic [1:0] token_startstop_internal;
    logic signed [NEW_TOKEN_BITS-1:0] tgt_new_good_tokens;
    logic signed [NEW_TOKEN_BITS-1:0] tgt_new_bad_tokens;
    logic signed [NEW_TOKEN_BITS-1:0] net_new_good_tokens, net_new_bad_tokens, proc_new_good_tokens, proc_new_bad_tokens;

    // instantiate the start/stop token buffer
    logic token_start_buf [NUM_PROCESSORS-1:0];
    logic token_stop_buf [NUM_PROCESSORS-1:0];

    logic [$clog2(NUM_PROCESSORS)-1:0] processor_id;
    logic [$clog2(NUM_PROCESSORS+1)-1:0] processor_id_internal, processor_id_internal_prev;


    // instantiate the network
    tt_um_jleugeri_ttt_network #(
        .NUM_PROCESSORS(NUM_PROCESSORS),
        .NUM_CONNECTIONS(NUM_CONNECTIONS),
        .NEW_TOKEN_BITS(NEW_TOKEN_BITS) 
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
        .prog_tokens(net_prog_tokens)
    );


    // instantiate the processor core
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
        .new_good_tokens(proc_new_good_tokens),
        .new_bad_tokens(proc_new_bad_tokens),
        // data outputs
        .token_startstop(token_startstop_internal),
        // programming inputs
        .instruction(instruction_proc),
        .prog_duration(proc_prog_duration),
        .prog_threshold(proc_prog_threshold)
    );

    logic first_cycle;

    always_ff @( posedge clock_fast ) begin
        if (reset) begin
            // next stage should be INPUT
            stage <= 2'b00;
        end
        else begin
            // if the MSB of instruction is high, we are in the programming mode
            // the second MSB in the programming mode determines whether we're configuring the network or the processors
            case (instruction[3:2])
                2'b11 : begin
                    // if the second MSB is set, configure the network
                    instruction_net <= {1'b1, instruction[1:0]};
                    instruction_proc <= 3'b000;
                    source_id <= processor_id_in;
                    connection_id <= connection_id_in;
                    net_prog_tokens <= prog_tokens;
                end
                // alternatively, configure the processor
                2'b10 : begin
                    instruction_net <= 3'b000;
                    instruction_proc <= {1'b1,instruction[1:0]};
                    processor_id <= $clog2(NUM_PROCESSORS)'(processor_id_in);
                    proc_prog_duration <= prog_duration;
                    proc_prog_threshold <= prog_threshold;
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
                        2'b00 : begin
                            // the two LSB of the instruction select the operation
                            case (instruction[1:0])
                                2'b00: begin
                                    // do nothing (block)
                                    processor_id <= 0;
                                    instruction_proc <= 3'b000;
                                    instruction_net <= 3'b000;
                                end

                                // read input
                                2'b01: begin
                                    instruction_proc <= 3'b001;
                                    processor_id <= $clog2(NUM_PROCESSORS)'(processor_id_in);
                                    proc_new_good_tokens <= good_tokens_in;
                                    proc_new_bad_tokens <= bad_tokens_in;
                                end

                                // advance to the next stage
                                2'b10: begin
                                    stage <= 2'b01;
                                    // set up for looping over all processors
                                    processor_id <= 0;
                                    processor_id_internal <= 0;
                                    processor_id_internal_prev <= -1;
                                    proc_new_good_tokens <= 0;
                                    proc_new_bad_tokens <= 0;
                                    instruction_proc <= 3'b010;
                                end

                                // RESERVED
                                default: begin
                                    // block!
                                end
                            endcase
                        end

                        // stage 2: update all processors
                        2'b01 : begin

                            // outputs of the processors arrive with a one-cycle delay
                            // if this was not the first clock cycle, store the output of the previous processor
                            if(processor_id_internal != 0) begin
                                token_start_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal_prev)] <= token_startstop_internal[1];
                                token_stop_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal_prev)] <= token_startstop_internal[0];
                            end

                            //advance to next stage
                            if (processor_id_internal == MAX_PROCESSOR_ID_PLUS_ONE) begin
                                stage <= 2'b10;
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
                                processor_id <= $clog2(NUM_PROCESSORS)'(processor_id_internal) + 1;
                            end
                        end

                        // stage 3
                        // go through all processors, check if they fired a token
                        2'b10 : begin
                            if (processor_id_internal == MAX_PROCESSOR_ID_PLUS_ONE) begin
                                // if we reached the end, go back to waiting for external input
                                stage <= 2'b00;
                            end
                            else begin
                                // cycle over all processors, and for each, cycle over all connections before moving on to the next processor
                                if (token_start_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal)] ^ token_stop_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal)]) begin
                                    // write output back to host
                                    processor_id_out <= $clog2(NUM_PROCESSORS)'(processor_id_internal);
                                    token_startstop <= {token_start_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal)], token_stop_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal)]};
                                    output_valid <= 1;
                                    // if the processor started/stopped a token, start cycling over its connections

                                    stage <= 2'b11;
                                    instruction_net <= 3'b010;
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

                        2'b11 : begin
                            // continue cycling
                            first_cycle <= 0;

                            token_startstop <= 2'b00;
                            output_valid <= 0;

                            // if we're done iterating, go to the next processor
                            if (network_done) begin
                                instruction_net <= 3'b000;
                                stage <= 2'b10;
                                instruction_proc <= 3'b000;
                            end 
                            else begin
                                instruction_net <= 3'b011;
                                if (!first_cycle) begin
                                    // if the processor started a token, use positive weights, otherwise, negative
                                    
                                    if (token_start_buf[$clog2(NUM_PROCESSORS)'(processor_id_internal_prev)]) begin
                                        proc_new_good_tokens <= net_new_good_tokens;
                                        proc_new_bad_tokens <= net_new_bad_tokens;
                                    end
                                    else begin
                                        proc_new_good_tokens <= -net_new_good_tokens;
                                        proc_new_bad_tokens <= -net_new_bad_tokens;
                                    end

                                    if (network_valid) begin
                                        processor_id <= target_id;
                                        instruction_proc <= 3'b001;
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
endmodule : tt_um_jleugeri_ttt_main
