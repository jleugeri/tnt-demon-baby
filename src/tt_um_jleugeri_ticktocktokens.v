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
    //assign uo_out[7:2] = 6'b111111;
    assign uio_out = 8'b11111111;

    tt_um_jleugeri_ttt_respite #(8) respite (
        .clk(clk),
        .in(ui_in),
        .nonempty(uo_out[2]),
        .current_idx(uo_out[7:5]),
        .current_val(uo_out[3]),
        .go(ena),
        .done(uo_out[4])
    );

    // instantiate the event processor
    tt_um_jleugeri_ttt_processor_core #(
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