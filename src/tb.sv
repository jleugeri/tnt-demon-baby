`default_nettype none
`timescale 1ns/1ps

/*
this testbench just instantiates the module and makes some convenient wires
that can be driven / tested by the cocotb test.py
*/

// testbench is controlled by test.py
module tb ();

    // this part dumps the trace to a vcd file that can be viewed with GTKWave
    initial begin
        //$dumpfile ("tb.vcd");
        $dumpvars (0, tb);
        #1;
    end

    // instantiate the sub-testbenches
    tb_processor_core tb_processor_core();
    tb_network tb_network();
    tb_main tb_main();
    tb_ticktocktokens tb_ticktocktokens();

endmodule : tb
