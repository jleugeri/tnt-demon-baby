module tt_um_jleugeri_ttt_network #(
    parameter int NUM_PROCESSORS,
    parameter int NUM_CONNECTIONS = NUM_PROCESSORS*NUM_PROCESSORS,
    parameter int NEW_TOKEN_BITS = 4
) (
    // control inputs / outputs
    input logic clk,
    input logic reset,
    input logic [$clog2(NUM_PROCESSORS)-1:0] processor_id,
    input logic [$clog2(NUM_CONNECTIONS)-1:0] connection_id,
    output logic done,
    output logic valid,
    
    // outputs to processor
    output logic [$clog2(NUM_PROCESSORS)-1:0] target_id,
    output logic signed [NEW_TOKEN_BITS-1:0] new_good_tokens,
    output logic signed [NEW_TOKEN_BITS-1:0] new_bad_tokens,
    
    // programming inputs
    input logic [2:0] instruction,
    input logic [NEW_TOKEN_BITS-1:0] prog_tokens
);

    // target loop counter
    logic [$clog2(NUM_CONNECTIONS)-1:0] tgt_addr;
    logic [$clog2(NUM_CONNECTIONS)-1:0] end_addr;

    // weight memory: for each processor, we store good and bad in csc format
    // the start indices of theses lists are stored in tgt_addr_first
    logic [$clog2(NUM_CONNECTIONS)-1:0] tgt_indptr[NUM_PROCESSORS:0];
    logic [$clog2(NUM_PROCESSORS)-1:0] tgt_indices[NUM_CONNECTIONS-1:0];
    logic [NEW_TOKEN_BITS-1:0] tgt_new_good_tokens[NUM_CONNECTIONS-1:0];
    logic [NEW_TOKEN_BITS-1:0] tgt_new_bad_tokens[NUM_CONNECTIONS-1:0];

    // memory address counters


    logic looping;

    always_ff @( posedge clk ) begin
        if (reset) begin
            looping <= 0;
        end 
        else begin
            case (instruction)
                // EXECUTION MODE INSTRUCTIONS

                // do nothing
                3'b000 : begin
                    done <= 0;
                    valid <= 0;
                end

                // RESERVED
                3'b001 : begin
                end

                // start iteration over all outgoing connections for this neuron
                3'b010 : begin
                    // stage 1: load the address range and start iterating
                    tgt_addr <= tgt_indptr[processor_id];
                    end_addr <= tgt_indptr[processor_id+1];
                    done <= 0;
                    valid <= 0;
                end

                // keep going until we have iterated over all outgoing connections for this neuron once, then wait
                3'b011 : begin
                    if (tgt_addr != end_addr) begin
                        // we now have valid data
                        target_id <= tgt_indices[tgt_addr];
                        new_good_tokens <= tgt_new_good_tokens[tgt_addr];
                        new_bad_tokens <= tgt_new_bad_tokens[tgt_addr];

                        tgt_addr <= tgt_addr + 1;
                        valid <= 1;
                    end
                    else begin
                        done <= 1;
                        valid <=0;
                    end
                end

                // PROGRAMMING MODE INSTRUCTIONS

                // set the good token weight for the currently selected connection
                3'b100 : begin
                    tgt_new_good_tokens[connection_id] <= prog_tokens;
                end

                // set the bad token weight for the currently selected connection
                3'b101 : begin
                    tgt_new_bad_tokens[connection_id] <= prog_tokens;
                end

                // set the indptr address for the currently selected processor
                3'b110: begin
                    tgt_indptr[processor_id] <= connection_id;
                end

                // set the index for the currently selected connection
                3'b111 : begin
                    tgt_indices[connection_id] <= processor_id;
                end
            endcase
        end
    end

endmodule: tt_um_jleugeri_ttt_network
