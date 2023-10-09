import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles


@cocotb.test()
async def test_ticktocktokens(dut):
    dut._log.info("start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # set good and bad token threshold to 2 and duration to maximum (15)
    dut.uio_in.value = 0b00101111
    
    dut._log.info("reset")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)

    assert dut.main.core.reset == 1

    dut.rst_n.value = 1

    await ClockCycles(dut.clk, 3)

    # trigger the first event by injecting one good token every step for four cycles
    dut.ui_in.value = 0b00000001
    await ClockCycles(dut.clk, 4)
    dut.ui_in.value = 0b00000000

    await ClockCycles(dut.clk, 2)
    
    # start subtracting tokens
    dut.ui_in.value = 0b00001111
    await ClockCycles(dut.clk, 4)

    # add and remove some more token with no effect
    dut.ui_in.value = 0b00000001
    await ClockCycles(dut.clk, 2)
    dut.ui_in.value = 0b00001111
    await ClockCycles(dut.clk, 2)
    dut.ui_in.value = 0b00000000

    await ClockCycles(dut.clk, 10)

    # trigger the second event by injecting one good token every step for four cycles
    dut.ui_in.value = 0b00000001
    await ClockCycles(dut.clk, 4)

    # now start injecting bad tokens to stop event
    dut.ui_in.value = 0b00010000
    await ClockCycles(dut.clk, 4)
    # now start removing bad tokens to potentially re-trigger event
    dut.ui_in.value = 0b11110000
    await ClockCycles(dut.clk, 4)
    
    # start subtracting good tokens again
    dut.ui_in.value = 0b00001111
    await ClockCycles(dut.clk, 4)
    dut.ui_in.value = 0b00000000

    await ClockCycles(dut.clk, 15)

    dut._log.info("Done!")
