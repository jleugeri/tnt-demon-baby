`default_nettype none

module tt_um_jleugeri_demon_baby #( parameter MAX_COUNT = 10_000_000 ) (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Bidirectional Input path
    output wire [7:0] uio_out,  // IOs: Bidirectional Output path
    output wire [7:0] uio_oe,   // IOs: Bidirectional Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // will go high when the design is enabled
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

    wire reset = ! rst_n;

    // set up direction of bidirectional IOs
    assign uio_oe = 8'b11111111;


    always @(posedge clk) begin
        // if reset, reset all internal registers
        if (reset) begin
        end else begin
        end
    end

    // instantiate submodules

endmodule
