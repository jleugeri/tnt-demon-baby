# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

# defaults
SIM ?= icarus 
#SIM ?= verilator
TOPLEVEL_LANG ?= verilog

TEST_SEPARATELY ?= no

# flags for verilator
ifeq ($(SIM),verilator)
COMPILE_ARGS	+= --timing
EXTRA_ARGS 		+= --trace --trace-structs
endif

# normal simulation
ifneq ($(GATES),yes)

# this is the only part you should need to modify:
WAVES 			 = 1


# $(PWD)/tt_um_jleugeri_ttt_network.v $(PWD)/tt_um_jleugeri_ttt_respite.v 
VERILOG_SOURCES += $(PWD)/tt_um_jleugeri_ttt_processor_core.sv $(PWD)/tt_um_jleugeri_ttt_network.sv $(PWD)/tt_um_jleugeri_ttt_main.sv $(PWD)/tt_um_jleugeri_ticktocktokens.sv

ifeq ($(TEST_SEPARATELY),yes)
VERILOG_SOURCES += $(PWD)/tb_all.sv $(PWD)/tb_processor_core.sv $(PWD)/tb_network.sv $(PWD)/tb_main.sv $(PWD)/tb_ticktocktokens.sv
else
VERILOG_SOURCES += $(PWD)/tb_ticktocktokens.sv
endif 

else

# gate level simulation requires some extra setup, you shouldn't need to touch this
COMPILE_ARGS    += -DGL_TEST
COMPILE_ARGS    += -DFUNCTIONAL
COMPILE_ARGS    += -DUSE_POWER_PINS
COMPILE_ARGS    += -DSIM
COMPILE_ARGS    += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v

# this gets copied in by the GDS action workflow
ifeq ($(TEST_SEPARATELY),yes)
VERILOG_SOURCES += $(PWD)/tb_all.sv $(PWD)/tb_processor_core.sv $(PWD)/tb_network.sv $(PWD)/tb_main.sv $(PWD)/tb_ticktocktokens.sv
else
VERILOG_SOURCES += $(PWD)/tb_ticktocktokens.sv
endif 

VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

endif

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
ifeq ($(TEST_SEPARATELY),yes)
TOPLEVEL = tb_all
# MODULE is the basename of the Python test file
MODULE = test_separately
else
TOPLEVEL = tb_ticktocktokens
# MODULE is the basename of the Python test file
MODULE = test
endif 


# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim