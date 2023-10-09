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

    wire reset = !rst_n;

    // set up direction of bidirectional IOs
    assign uio_oe = 8'b00000000;
    assign uo_out[7:2] = 6'b111111;
    assign uio_out = 8'b11111111;

    // instantiate the event processor
    tt_um_jleugeri_event_processor_core #(
        .NEW_TOKENS_BITS(4),
        .TOKENS_BITS(4),
        .DURATION_BITS(4)
    ) core (
        .reset(reset),
        .clock_fast(clk),
        .clock_slow(clk),
        .new_good_tokens(ui_in[3:0]),
        .new_bad_tokens(ui_in[7:4]),
        .token_start(uo_out[0]),
        .token_end(uo_out[1]),
        .good_tokens_threshold(uio_in[7:4]),
        .bad_tokens_threshold(uio_in[7:4]),
        .duration(uio_in[3:0])
    );

endmodule

module tt_um_jleugeri_event_processor_core #(
    parameter NEW_TOKENS_BITS = 4,
    parameter TOKENS_BITS = 8,
    parameter DURATION_BITS = 4
) (
    // control inputs
    input logic clock_fast,
    input logic clock_slow,
    input logic reset,
    // data inputs
    input logic signed [NEW_TOKENS_BITS-1:0] new_good_tokens,
    input logic signed [NEW_TOKENS_BITS-1:0] new_bad_tokens,
    // data outputs
    output logic token_start,
    output logic token_end,
    // configuration inputs
    input logic [TOKENS_BITS-1:0] good_tokens_threshold,
    input logic [TOKENS_BITS-1:0] bad_tokens_threshold,
    input logic [DURATION_BITS-1:0] duration
);

    // internal state variables
    logic signed [TOKENS_BITS-1:0] good_tokens;
    logic signed [TOKENS_BITS-1:0] bad_tokens;
    logic [DURATION_BITS-1:0] remaining_duration;


    logic isOn;
    logic setOn;
    logic setOff;

    logic timerExpired;

    // two-way handshake for clock domain crossing
    logic startCountdown;
    logic startedCountdown;

    always_comb begin
        // timer is expired if the remaining duration is zero
        timerExpired = (remaining_duration == 0);

        // combinatorial logic to determine when to set the outputs
        setOn = (good_tokens >= 0) && (bad_tokens <= 0) && !isOn;
        //setOff = ((good_tokens < 0) && (bad_tokens > 0) && isOn) || (timerExpired && isOn);
        setOff = ((bad_tokens > 0) && isOn) || (timerExpired && isOn);
    end

    always_ff @(posedge clock_fast ) begin
        // if reset, reset all internal registers
        if (reset) begin
            remaining_duration <= 0;
            // initialize the token counts to negative thresholds
            good_tokens <= -good_tokens_threshold;
            bad_tokens <= -bad_tokens_threshold;
        end else begin    
            // on positive edge of fast clock, count the inputs
            good_tokens <= good_tokens + TOKENS_BITS'(new_good_tokens);
            bad_tokens <= bad_tokens + TOKENS_BITS'(new_bad_tokens);
        end
    end

    always_ff @(negedge clock_fast) begin
        if (!reset) begin
            // on negative clock edge, turn own token & time-out on or off
            if (setOn) begin
                // turn on 
                isOn <= 1;

                // initialize the countdown
                startCountdown <= 1;

                // signal the beginning of the token
                token_start <= 1;
                token_end <= 0;
            end else if (setOff) begin
                // turn off
                isOn <= 0;

                // dont initialize the countdown
                startCountdown <= 0;

                // signal the end of the token
                token_end <= 1;
                token_start <= 0;
            end else begin
                // reset the outputs
                token_start <= 0;
                token_end <= 0;

                // part one of two-way handshake for clock domain crossing:
                // the countdown was started, so we don't want to start it again
                if (startedCountdown) begin
                    startCountdown <= 0;
                end
            end


        end
    end

    always_ff @(posedge clock_slow) begin
        if (!reset) begin
            if (isOn && startCountdown && !startedCountdown) begin
                remaining_duration <= duration;
                startedCountdown <= 1;
            end else if (isOn && !timerExpired) begin
               // if the countdown is running, decrement on positive edge of slow clock
                remaining_duration <= remaining_duration - 1;
                startedCountdown <= 0;
            end else begin
                // part two of two-way handshake for clock domain crossing:
                // the countdown's start has been acknowledged, so we can stop signalling it
                if (!startedCountdown) begin
                    startedCountdown <= 0;
                end
            end
        end
    end

endmodule