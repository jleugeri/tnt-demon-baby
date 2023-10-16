package tt_um_jleugeri_ttt;
    /*
    Stages:
    - RESET: reset, optionally re-program the processor
    - INPUT: clear input token buffers and initialize with external input tokens
    - RECURRENT: add recurrently generated tokens to the input token buffers
    - UPDATE: update the processors by one step
    - OUTPUT: output the tokens that are ready to be output

    When the processor is ready to move to the next processing stage, it will assert the done signal.
    If the externally controlled hold signal is asserted, the processor will not move to the next stage until the hold signal is deasserted.
    When reset is high, the processor will immediately return to stage 0b00, and possibly (partially) reprogram.
    */
    typedef enum stage_t { RESET, INPUT, RECURRENT, UPDATE, OUTPUT } name;


endpackage : tt_um_jleugeri_ttt

module tt_um_jleugeri_ttt_main #(
    parameter int DATA_IN_BITS = 8
    parameter int DATA_OUT_BITS = 8
) (
    // control flow logic
    input  logic reset,
    input  logic clock_fast,
    input  logic clock_slow,
    input  logic hold_i,
    input  logic hold_o,
    output logic done,
    output stage_t stage,
    // data I/O logic
    input  logic [DATA_IN_BITS-1:0] data_in,
    output logic [DATA_OUT_BITS-1:0] data_out,
);
    // internal address wires
    wire [DURATION_BITS-1:0] idx_src;
    wire [DURATION_BITS-1:0] idx_tgt;

    // internal data wires
    wire src_tstart;
    wire src_tstop;
    wire signed [NEW_TOKENS_BITS-1:0] tgt_new_good_tokens;
    wire signed [NEW_TOKENS_BITS-1:0] tgt_new_bad_tokens;

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
