module tt_um_jleugeri_ttt_network #(
    parameter int NUM_PROCESSORS,
    parameter int NUM_CONNECTIONS = NUM_PROCESSORS*NUM_PROCESSORS,
    parameter int NEW_TOKENS_BITS = 4,
    parameter int PROG_WIDTH = 8
) (
    // control inputs / outputs
    input logic clk,
    input logic reset,
    input logic [$clog2(NUM_PROCESSORS)-1:0] source_id,
    input logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id,
    output logic done,
    
    // outputs to processor
    output logic [$clog2(NUM_PROCESSORS)-1:0] target_id,
    output logic signed [NEW_TOKENS_BITS-1:0] new_good_tokens,
    output logic signed [NEW_TOKENS_BITS-1:0] new_bad_tokens,
    
    // programming inputs
    input logic [2:0] instruction,
    input logic [PROG_WIDTH-1:0] prog_data
);

    // target loop counter
    logic [$clog2(NUM_CONNECTIONS)-1:0] tgt_addr;
    logic [$clog2(NUM_CONNECTIONS)-1:0] end_addr;

    // weight memory: for each processor, we store good and bad in csc format
    // the start indices of theses lists are stored in tgt_addr_first
    logic [$clog2(NUM_CONNECTIONS)-1:0] tgt_indptr[NUM_PROCESSORS:0];
    logic [$clog2(NUM_PROCESSORS)-1:0] tgt_indices[NUM_CONNECTIONS-1:0];
    logic [NEW_TOKENS_BITS-1:0] tgt_new_good_tokens[NUM_CONNECTIONS-1:0];
    logic [NEW_TOKENS_BITS-1:0] tgt_new_bad_tokens[NUM_CONNECTIONS-1:0];

    // memory address counters
    //logic [$clog2(NUM_PROCESSORS)]


    logic looping;

    always_ff @( posedge clk ) begin
        if (reset) begin
            looping <= 0;
        end 
        else begin
            case (instruction)
                // RESERVED
                3'b001 : begin
                end

                // set the indptr address for the currently selected processor
                3'b010: begin
                    tgt_indptr[source_id] <= prog_data[$clog2(NUM_CONNECTIONS)-1:0];
                end

                // set the index for the currently selected connection
                3'b011 : begin
                    tgt_indices[connection_id] <= prog_data[$clog2(NUM_PROCESSORS)-1:0];
                end

                // set the good token weight for the currently selected connection
                3'b100 : begin
                    tgt_new_good_tokens[connection_id] <= prog_data[NEW_TOKENS_BITS-1:0];
                end

                // set the bad token weight for the currently selected connection
                3'b101 : begin
                    tgt_new_bad_tokens[connection_id] <= prog_data[NEW_TOKENS_BITS-1:0];
                end

                // start iteration over all outgoing connections for this neuron
                3'b110 : begin
                    // stage 1: load the address range and start iterating
                    tgt_addr <= tgt_indptr[source_id];
                    end_addr <= tgt_indptr[source_id+1];
                    done <= 0;
                end

                // keep going until we have iterated over all outgoing connections for this neuron once, then wait
                3'b111 : begin
                    if (tgt_addr != end_addr) begin
                        // we now have valid data
                        target_id <= tgt_indices[tgt_addr];
                        new_good_tokens <= tgt_new_good_tokens[tgt_addr];
                        new_bad_tokens <= tgt_new_bad_tokens[tgt_addr];

                        tgt_addr <= tgt_addr + 1;
                    end
                    else begin
                        done <= 1;
                    end
                end

                // default: do nothing
                default : begin
                end
            endcase
        end
    end

endmodule: tt_um_jleugeri_ttt_network