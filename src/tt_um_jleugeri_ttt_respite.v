// *Re*cursive *Sp*arse *Ite*rator

module tt_um_jleugeri_ttt_respite #(
    parameter int SIZE
) (
    input logic clk,
    input logic [SIZE-1:0] in,
    output logic nonempty,
    // iteration variables
    output logic [$clog2(SIZE)-1:0] current_idx,
    output logic current_val,
    input logic go,
    output logic done
);

    // wire up the two subtrees
    logic nonempty_low, nonempty_high, current_val_low, current_val_high, go_low, go_high, done_low, done_high;

    // the non-empty property is immediately propagated
    assign nonempty = nonempty_low || nonempty_high;

    if (SIZE == 2) begin
        assign nonempty_low = in[0] != 0;
        assign nonempty_high = in[1] != 0;
        assign current_val_low = in[0];
        assign current_val_high = in[1];
        // current_idx_low[0] = 0;
        // current_idx_high[0] = 1;
        assign done_low = 1;
        assign done_high = 1;

    end else begin
        logic [$clog2(SIZE/2)-1:0] current_idx_low, current_idx_high;
        
        // low half
        tt_um_jleugeri_ttt_respite #(SIZE/2) bv_low (
            .clk(clk),
            .in(in[SIZE/2-1:0]),
            .nonempty(nonempty_low),
            .current_idx(current_idx_low),
            .current_val(current_val_low),
            .go(go_low),
            .done(done_low)
        );

        // high half
        tt_um_jleugeri_ttt_respite #(SIZE/2) bv_high (
            .clk(clk),
            .in(in[SIZE-1:SIZE/2]),
            .nonempty(nonempty_high),
            .current_idx(current_idx_high),
            .current_val(current_val_high),
            .go(go_high),
            .done(done_high)
        );

    end


endmodule