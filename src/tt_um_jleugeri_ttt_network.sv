module tt_um_jleugeri_ttt_network #(
    parameter int NUM_PROCESSORS,
    parameter int NUM_CONNECTIONS = NUM_PROCESSORS*NUM_PROCESSORS,
    parameter int NEW_TOKENS_BITS = 4,
    parameter int PROG_WIDTH = 8
) (
    // control inputs / outputs
    input logic clk,
    input logic reset,
    input logic valid_in,
    input logic [$clog2(NUM_PROCESSORS)-1:0] source_id,
    output logic done,
    output logic valid_out,

    // inputs from processor
    input logic [1:0] token_startstop,
    
    // outputs to processor
    output logic [$clog2(NUM_PROCESSORS)-1:0] target_id,
    output logic signed [NEW_TOKENS_BITS-1:0] new_good_tokens,
    output logic signed [NEW_TOKENS_BITS-1:0] new_bad_tokens,
    
    // programming inputs
    input logic [2:0] prog_header,
    input logic [PROG_WIDTH-1:0] prog_data
);

    // target loop counter
    logic [$clog2(NUM_CONNECTIONS)-1:0] tgt_addr;
    logic [$clog2(NUM_CONNECTIONS)-1:0] end_addr;

    // weight memory: for each processor, we store good and bad in csc format
    // the start indices of theses lists are stored in tgt_addr_first
    logic [$clog2(NUM_CONNECTIONS)-1:0] tgt_indptr[NUM_PROCESSORS:0];
    logic [$clog2(NUM_PROCESSORS)-1:0] tgt_indices[NUM_CONNECTIONS-1:0];
    logic [NEW_TOKENS_BITS-1:0] tgt_data_good[NUM_CONNECTIONS-1:0];
    logic [NEW_TOKENS_BITS-1:0] tgt_data_bad[NUM_CONNECTIONS-1:0];

    logic cycle_complete;

    always_ff @( posedge clk ) begin
        if (reset) begin
            cycle_complete <= 1;
            valid_out <= 0;
        end 
        else if(valid_in) begin
            // stage 1: load the weight memory and start iterating
            if (cycle_complete) begin
                cycle_complete <= 0;
                tgt_addr <= tgt_indptr[source_id];
                end_addr <= tgt_indptr[source_id+1]-1;
                valid_out <= 0;
            end
            else begin
                // stage 2: each cycle we update one connection

                // we now have valid data
                valid_out <= 1;
                target_id <= tgt_indices[tgt_addr];
                new_good_tokens <= tgt_data_good[tgt_addr];
                new_bad_tokens <= tgt_data_bad[tgt_addr];

                // move on to the next connection
                if (tgt_addr == end_addr) begin
                    cycle_complete <= 1;
                end 
                else begin
                    tgt_addr <= tgt_addr + 1;
                end
            end
        end
    end

endmodule: tt_um_jleugeri_ttt_network