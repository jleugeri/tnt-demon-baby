

module tt_um_jleugeri_ttt_processor_core #(
    parameter NEW_TOKENS_BITS = 4,
    parameter TOKENS_BITS = 8,
    parameter DURATION_BITS = 8,
    parameter NUM_PROCESSORS = 10,
    parameter PROG_WIDTH = 8
) (
    // control inputs
    input logic clock_fast,
    input logic clock_slow,
    input logic reset,
    input logic [$clog2(NUM_PROCESSORS)-1:0] processor_id,
    // data inputs
    input logic signed [NEW_TOKENS_BITS-1:0] new_good_tokens,
    input logic signed [NEW_TOKENS_BITS-1:0] new_bad_tokens,
    // data outputs
    output logic [1:0] token_startstop,
    // programming inputs
    input logic [2:0] instruction,
    input logic [PROG_WIDTH-1:0] prog_data
);
    // parameter memory
    logic [NUM_PROCESSORS-1:0][TOKENS_BITS-1:0] good_tokens_threshold;
    logic [NUM_PROCESSORS-1:0][TOKENS_BITS-1:0] bad_tokens_threshold;
    logic [NUM_PROCESSORS-1:0][DURATION_BITS-1:0] duration;

    // internal state variables
    logic signed [NUM_PROCESSORS-1:0][TOKENS_BITS-1:0] good_tokens;
    logic signed [NUM_PROCESSORS-1:0][TOKENS_BITS-1:0] bad_tokens;
    logic [NUM_PROCESSORS-1:0][DURATION_BITS-1:0] remaining_duration;
    logic [NUM_PROCESSORS-1:0] isOn;

    always_ff @(posedge clock_fast ) begin
        // if reset, reset all internal registers
        if (reset) begin
            // initialize the token counts to negative thresholds
            bad_tokens[processor_id] <= -bad_tokens_threshold[processor_id];
            good_tokens[processor_id] <= -good_tokens_threshold[processor_id];
            isOn[processor_id] <= 0;
            remaining_duration[processor_id] <= 0;
        end
        // otherwise, perform an action (either programming or running the processor)
        else begin
            // check if we should program the memory, and if so, which
            case (instruction)
                // DO NOTHING
                3'b000 : begin
                end

                // program the duration
                3'b001 : begin
                    duration[processor_id] <= prog_data;
                end
                // program the good tokens threshold
                3'b010 : begin
                    good_tokens_threshold[processor_id] <= prog_data;
                    good_tokens[processor_id] <= -prog_data;
                end
                // program the bad tokens threshold
                3'b011 : begin
                    bad_tokens_threshold[processor_id] <= prog_data;
                    bad_tokens[processor_id] <= -prog_data;
                end
                // update incoming tokens
                3'b100 : begin 
                    // pipelining step 1: update the neuron's counter
                    good_tokens[processor_id] <= good_tokens[processor_id] + TOKENS_BITS'(new_good_tokens);
                    bad_tokens[processor_id] <= bad_tokens[processor_id] + TOKENS_BITS'(new_bad_tokens);
                end

                // update the internal state
                3'b101 : begin
                    // pipelining step 2: check if we need to generate our own token here
                    if ( !isOn[processor_id] && ( good_tokens[processor_id] >= 0 ) && ( bad_tokens[processor_id] <= 0 ) ) begin
                        // turn on 
                        isOn[processor_id] <= 1;

                        // initialize the countdown                
                        remaining_duration[processor_id] <= duration[processor_id];

                        // signal the beginning of the token
                        token_startstop <= 2'b10;
                    end 
                    else if ( isOn[processor_id] && ((bad_tokens[processor_id] > 0) || remaining_duration[processor_id] == 0 )) begin
                        // turn off
                        isOn[processor_id] <= 0;

                        // end the countdown
                        remaining_duration[processor_id] <= 0;

                        // signal the end of the token
                        token_startstop <= 2'b01;
                    end 
                    else begin
                        // if the countdown is running and the slow clock is currently on, decrement the countdown
                        if (isOn[processor_id] && clock_slow) begin
                            remaining_duration[processor_id] <= remaining_duration[processor_id] - 1;
                        end

                        // reset the outputs
                        token_startstop <= 2'b00;
                    end
                end
                // default: do nothing
                default: begin
                end
            endcase
        end
    end

endmodule : tt_um_jleugeri_ttt_processor_core