import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles


@cocotb.test()
async def test_demon_baby(dut):
    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut._log.info("reset")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # dut._log.info("check all segments")
    # for i in range(10):
    #     dut._log.info("check segment {}".format(i))
    #     await ClockCycles(dut.clk, 1000)
    #     assert int(dut.segments.value) == segments[i]

    #     # all bidirectionals are set to output
    #     assert dut.uio_oe == 0xFF