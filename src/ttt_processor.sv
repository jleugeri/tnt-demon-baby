

module ttt_processor #(
    parameter NEW_TOKEN_BITS = 4,
    parameter TOKEN_BITS = 8,
    parameter DURATION_BITS = 8,
    parameter DATA_BITS = 8,
    parameter INSTRUCTION_BITS = 3
) (
    // control inputs
    input logic clock,
    input logic reset,
    input logic enable,
    input logic [INSTRUCTION_BITS-1:0] instruction,
    // data inputs
    input logic signed [NEW_TOKEN_BITS-1:0] good_tokens_in,
    input logic signed [NEW_TOKEN_BITS-1:0] bad_tokens_in,
    // data outputs
    output logic token_start,
    output logic token_stop,
    output logic token_valid,
    // programming inputs
    input logic [DATA_BITS-1:0] data_in,
    output logic [DATA_BITS-1:0] data_out
);

    // parameters
    logic [TOKEN_BITS-1:0] good_tokens_threshold;
    logic [TOKEN_BITS-1:0] bad_tokens_threshold;
    logic [DURATION_BITS-1:0] duration;

    // internal state variables
    logic signed [TOKEN_BITS-1:0] good_tokens;
    logic signed [TOKEN_BITS-1:0] bad_tokens;
    logic [DURATION_BITS-1:0] remaining_duration;
    logic isOn;

    always_ff @(posedge clock) begin
        // if reset, reset all internal registers
        if (reset) begin
            // initialize the token counts to negative thresholds
            bad_tokens <= 0;
            good_tokens <= 0;
            isOn <= 0;
            remaining_duration <= 0;
            token_start <= 0;
            token_stop <= 0;
            token_valid <= 0;
        end
        // otherwise, perform an action (either programming or running the processor)
        else if (enable) begin
            // check what we should do
            case (instruction)
                // 4'b0000: add/substract good tokens
                4'b0000 : begin
                    good_tokens <= good_tokens + TOKEN_BITS'(good_tokens_in);
                    data_out <= DATA_BITS'(good_tokens_in);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b0001: add/substract bad tokens
                4'b0001 : begin
                    bad_tokens <= bad_tokens + TOKEN_BITS'(bad_tokens_in);
                    data_out <= DATA_BITS'(bad_tokens_in);
                    token_start <= 0;
                    token_stop <= 0;
                end

                // 4'b0010: set good token count
                4'b0010 : begin
                    good_tokens <= data_in[TOKEN_BITS-1:0];
                    data_out <= DATA_BITS'(data_in[TOKEN_BITS-1:0]);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b0011: get good token count
                4'b0011 : begin
                    data_out <= DATA_BITS'(good_tokens);
                    token_start <= 0;
                    token_stop <= 0;
                end

                // 4'b0100: set bad token count
                4'b0100 : begin
                    bad_tokens <= data_in[TOKEN_BITS-1:0];
                    data_out <= DATA_BITS'(data_in[TOKEN_BITS-1:0]);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b0101: get bad token count
                4'b0101 : begin
                    data_out <= DATA_BITS'(bad_tokens);
                    token_start <= 0;
                    token_stop <= 0;
                end

                // 4'b0110: set remaining duration (implicitly sets on-state)
                4'b0110 : begin
                    remaining_duration <= data_in[DURATION_BITS-1:0];
                    if (data_in[DURATION_BITS-1:0] == 0) begin
                        isOn <= 0;
                    end
                    else begin
                        isOn <= 1;
                    end
                    data_out <= DATA_BITS'(data_in[DURATION_BITS-1:0]);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b0111: get remaining duration (implicitly gets on-state)
                4'b0111 : begin
                    data_out <= DATA_BITS'(remaining_duration);
                    token_start <= 0;
                    token_stop <= 0;
                end

                // 4'b1000: Tally up and make a decision (start/stop token?)
                4'b1000 : begin
                    if ( !isOn && ( good_tokens >= good_tokens_threshold ) && ( bad_tokens <= bad_tokens_threshold ) ) begin
                        // turn on 
                        isOn <= 1;

                        // initialize the countdown                
                        remaining_duration <= duration;

                        // signal the beginning of the token
                        token_start <= 1;
                        token_stop <= 0;
                    end 
                    else if ( isOn && (remaining_duration == 0)  && ( good_tokens >= good_tokens_threshold ) && ( bad_tokens <= bad_tokens_threshold )) begin
                        // we'd just turn off and on again, so instead, just stay on
                        // restart countdown timer
                        remaining_duration <= duration;
                    end
                    else if ( isOn && ((bad_tokens > bad_tokens_threshold) || remaining_duration == 0 )) begin
                        // turn off
                        isOn <= 0;

                        // end the countdown
                        remaining_duration <= 0;

                        // signal the end of the token
                        token_stop <= 1;
                        token_start <= 0;
                    end 
                    else begin
                        // reset the outputs
                        token_stop <= 0;
                        token_start <= 0;
                    end
                end

                // 4'b1001: advance countdown
                4'b1001 : begin
                    if (isOn && remaining_duration != 0)  begin
                        remaining_duration <= remaining_duration - 1;
                    end
                    token_start <= 0;
                    token_stop <= 0;
                end


                // 4'b1010: set good token threshold
                4'b1010 : begin
                    good_tokens_threshold <= data_in[TOKEN_BITS-1:0];
                    data_out <= DATA_BITS'(data_in[TOKEN_BITS-1:0]);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b1011: get good token threshold
                4'b1011 : begin
                    data_out <= DATA_BITS'(good_tokens_threshold);
                    token_start <= 0;
                    token_stop <= 0;
                end


                // 4'b1100: set bad token threshold
                4'b1100 : begin
                    bad_tokens_threshold <= data_in[TOKEN_BITS-1:0];
                    data_out <= DATA_BITS'(data_in[TOKEN_BITS-1:0]);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b1101: get bad token threshold
                4'b1101 : begin
                    data_out <= DATA_BITS'(bad_tokens_threshold);
                    token_start <= 0;
                    token_stop <= 0;
                end


                // 4'b1110: set token duration
                4'b1110 : begin
                    duration <= data_in[DURATION_BITS-1:0];
                    data_out <= DATA_BITS'(data_in[DURATION_BITS-1:0]);
                    token_start <= 0;
                    token_stop <= 0;
                end
                
                // 4'b1111: get token duration
                4'b1111 : begin
                    data_out <= DATA_BITS'(duration);
                    token_start <= 0;
                    token_stop <= 0;
                end
            endcase
        end
    end

endmodule : ttt_processor
