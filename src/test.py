from typing import Union, Optional
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles
from cocotb.utils import get_sim_time
import numpy as np
from scipy import sparse as sps
from ttt_pyttt import PyTTT
from utils import *

async def program_processor(
        clock, 
        dut, 
        goodTokenThreshold: Optional[int] = None, 
        badTokenThreshold: Optional[int] = None, 
        duration: Optional[int] = None, 
        goodTokenCount: Optional[int] = None,
        badTokenCount: Optional[int] = None,
        remainingDuration: Optional[int] = None,
        flags: int = 0b0000):
    dut._log.info("programming core...")

    assert dut.rst_n.value == 1, "Cannot program during reset!"

    if duration is not None:
        # set the duration to DURATION
        dut.ui_in.value  = 0b1110 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(duration)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == duration, "duration not programmed correctly (observed {} != {})".format(dut.uo_out.value, duration)

    if goodTokenThreshold is not None:
        # set the good token threshold
        dut.ui_in.value  = 0b1010 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(goodTokenThreshold)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == goodTokenThreshold, "good token threshold not programmed correctly (observed {} != {})".format(dut.uo_out.value, goodTokenThreshold)

    if badTokenThreshold is not None:
        # set the bad token threshold
        dut.ui_in.value  = 0b1100 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(badTokenThreshold)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == badTokenThreshold, "bad token threshold not programmed correctly (observed {} != {})".format(dut.uo_out.value, badTokenThreshold)
    
    if goodTokenCount is not None:
        # set the good token count
        dut.ui_in.value  = 0b0010 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(goodTokenCount)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == goodTokenCount, "good token count not programmed correctly (observed {} != {})".format(dut.uo_out.value, goodTokenCount)

    if badTokenCount is not None:
        # set the bad token count
        dut.ui_in.value  = 0b0100 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(badTokenCount)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == badTokenCount, "bad token count not programmed correctly (observed {} != {})".format(dut.uo_out.value, badTokenCount)

    if remainingDuration is not None:
        # set the remaining duration
        dut.ui_in.value  = 0b0110 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(remainingDuration)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == remainingDuration, "remaining duration not programmed correctly (observed {} != {})".format(dut.uo_out.value, remainingDuration)

    # Clean up again
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(clock, 1)

async def get_processor_state(
        clock, 
        dut,
        goodTokenThreshold: Union[bool, int] = True, 
        badTokenThreshold: Union[bool, int] = True, 
        duration: Union[bool, int] = True, 
        goodTokenCount: Union[bool, int] = True,
        badTokenCount: Union[bool, int] = True,
        remainingDuration: Union[bool, int] = True):
    
    dut._log.info("getting processor state...")

    assert dut.rst_n.value == 1, "Cannot read processor state during reset!"

    observed = {}

    if duration is not False:
        # get the duration
        dut.ui_in.value  = 0b1111
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["duration"] = dut.uo_out.value.integer
        if not isinstance(duration, bool):
            assert dut.uo_out.value.integer == duration, "duration not programmed correctly (observed {} != {})".format(dut.uo_out.value, duration)

    if goodTokenThreshold is not False:
        # get the good token threshold
        dut.ui_in.value  = 0b1010
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["goodTokenThreshold"] = dut.uo_out.value.integer
        if not isinstance(goodTokenThreshold, bool):
            assert dut.uo_out.value.integer == goodTokenThreshold, "good token threshold not programmed correctly (observed {} != {})".format(dut.uo_out.value, goodTokenThreshold)

    if badTokenThreshold is not False:
        # get the bad token threshold
        dut.ui_in.value  = 0b1100
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["badTokenThreshold"] = dut.uo_out.value.integer
        if not isinstance(badTokenThreshold, bool):
            assert dut.uo_out.value.integer == badTokenThreshold, "bad token threshold not programmed correctly (observed {} != {})".format(dut.uo_out.value, badTokenThreshold)
    
    if goodTokenCount is not False:
        # get the good token count
        dut.ui_in.value  = 0b0010
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["goodTokenCount"] = dut.uo_out.value.integer
        if not isinstance(goodTokenCount, bool):
            assert dut.uo_out.value.integer == goodTokenCount, "good token count not programmed correctly (observed {} != {})".format(dut.uo_out.value, goodTokenCount)

    if badTokenCount is not False:
        # get the bad token count
        dut.ui_in.value  = 0b0100
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["badTokenCount"] = dut.uo_out.value.integer
        if not isinstance(badTokenCount, bool):
            assert dut.uo_out.value.integer == badTokenCount, "bad token count not programmed correctly (observed {} != {})".format(dut.uo_out.value, badTokenCount)

    if remainingDuration is not False:
        # get the remaining duration
        dut.ui_in.value  = 0b0110
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["remainingDuration"] = dut.uo_out.value.integer
        if not isinstance(remainingDuration, bool):
            assert dut.uo_out.value.integer == remainingDuration, "remaining duration not programmed correctly (observed {} != {})".format(dut.uo_out.value, remainingDuration)

    # Clean up again
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(clock, 1)

@cocotb.test()
async def test_ticktocktoken_programming(dut):
    NUM_SAMPLES = 100

    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk

    await ClockCycles(clock, 3)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    dut.ena.value = 1
    await ClockCycles(clock, 1)

    # generate parameters
    goodTokenThreshold = int(np.random.poisson(1))
    badTokenThreshold = int(np.random.poisson(0.25))
    duration = int(np.random.poisson(5))

    # generate initial conditions
    goodTokenCount = int(np.random.poisson(1))
    badTokenCount = int(np.random.poisson(0.25))
    remainingDuration = int(np.random.poisson(5))

    # program the simulated hardware
    await program_processor(dut.clk, dut, 
                            goodTokenThreshold=goodTokenThreshold, 
                            badTokenThreshold=badTokenThreshold, 
                            duration=duration,
                            goodTokenCount=goodTokenCount,
                            badTokenCount=badTokenCount,
                            remainingDuration=remainingDuration)

    # get processor state
    state = await get_processor_state(dut.clk, dut)


@cocotb.test(skip=True)
async def test_ticktocktoken_execution(dut):
    NUM_PROCESSORS = int(dut.ttt.NUM_PROCESSORS)
    NUM_CONNECTIONS = int(dut.ttt.NUM_CONNECTIONS)
    assert NUM_PROCESSORS > 0, "NUM_PROCESSORS must be greater than 0"
    assert NUM_CONNECTIONS > 0, "NUM_CONNECTIONS must be greater than 0"

    NUM_SAMPLES = 100

    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk

    await ClockCycles(clock, 3)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1

    # reset every neuron
    timeout = True
    for i in range(1_000_000):
        await ClockCycles(clock, 1)

        if(dut.uo_out.value[6:7] == 0b00):
            timeout = False
            break

    assert not timeout, "timeout while resetting"
    await ClockCycles(clock, 1)

    # generate the weights
    while True:
        W_good = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        W_bad = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        
        # make sure the first processor has outgoing connections
        W_good[1 % NUM_PROCESSORS,0] = 1
        W_bad[2 % NUM_PROCESSORS,0] = 1

        has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))

        if has_connection.nnz <= NUM_CONNECTIONS and has_connection.nnz > 10 and W_good.max() < 5 and W_bad.max() < 5:
            break

    # generate parameters
    goodTokenThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    badTokenThreshold = np.random.poisson(0.25, (NUM_PROCESSORS,)).astype(int)
    duration = np.random.poisson(5, (NUM_PROCESSORS,)).astype(int)

    # make sure that processor 0, which is written first, is excitable
    goodTokenThreshold[0] = 1
    badTokenThreshold[0] = 1
    duration[0] = 2

    # program the simulated hardware
    await program_via_interface(dut.clk, dut, goodTokenThreshold, badTokenThreshold, W_good, W_bad, duration)


    # create a golden reference model
    golden = PyTTT(goodTokenThreshold, badTokenThreshold, W_good, W_bad, duration)

    # generate some random input
    my_good_tokens_in = np.random.poisson(1, size=(NUM_SAMPLES, NUM_PROCESSORS)).astype(int)
    my_bad_tokens_in = np.random.poisson(0.1, size=(NUM_SAMPLES, NUM_PROCESSORS)).astype(int)

    # make sure the first processor fires in the first step
    my_good_tokens_in[0,0] = 1
    my_bad_tokens_in[0,0] = 0

    # give each incoming token a lifetime of DELAY
    PyTTT.set_expiration(my_good_tokens_in, 10)
    PyTTT.set_expiration(my_bad_tokens_in, 10)

    # run both implementations and compare the results
    all_should_startstop = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=object)
    all_did_startstop = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=object)
    times = np.zeros((NUM_SAMPLES, ), dtype=np.int64)

    for i in range(NUM_SAMPLES):
        for j in range(NUM_PROCESSORS):
            all_did_startstop[i,j] = (False, False)

    # first, record the reference signals
    for (step, (should_start, should_stop)) in enumerate(golden.run(my_good_tokens_in, my_bad_tokens_in)):
        all_should_startstop[step,:] = list(zip(should_start, should_stop))

    assert all_should_startstop[0,0] == (True, False), "first processor did not start in first step, but we programmed it to do so!"

    # then record our hardware implementation
    for (step,(good_in, bad_in)) in enumerate(zip(my_good_tokens_in, my_bad_tokens_in)):
        #print("Step {}".format(step))
        nz = np.nonzero(np.logical_or(good_in != 0, bad_in != 0))[0]

        if (len(nz) == 0):
            # proceed directly to next stage, because there is no input
            dut.ui_in.value = 0b00100000
        else:
            # hold execution due to external input

            # present all the good and bad inputs one by one
            for idx in nz:
                dut.ui_in.value = (0b0001 << 4) | to_n_bit_twos_complement(int(idx),4)
                dut.uio_in.value = (to_n_bit_twos_complement(int(good_in[idx]),4) << 4) | to_n_bit_twos_complement(int(bad_in[idx]),4)

                await ClockCycles(clock, 1)

            dut.uio_in.value = 0

            # release execution            
            dut.ui_in.value = 0b00100000
        
        # wait for at least one cycle
        await ClockCycles(clock, 1)
        # then assert the instruction 0b0000 to block when re-entering input stage
        dut.ui_in.value = 0b00000000

        timeout = True
        for i in range(1_000_000):
            await FallingEdge(clock)

            #print(dut.stage.value, dut.output_valid.value, dut.token_startstop.value.binstr)

            data_out = dut.uo_out.value
            # unfortunately, this gets sliced wrong in cocotb, so we need to reverse everything
            #proc_out = data_out[7:4]
            #stsp = data_out[3:2]
            #stat = data_out[1:0]
            proc_out = data_out[0:3]
            stsp = data_out[4:5]
            stat = data_out[6:7]
            if (stsp != 0b00):
                all_did_startstop[step,proc_out] = tuple(map(bool, stsp))
                times[step] = get_sim_time("ns")

            if (stat.value == 0):
                timeout = False
                break
        
        await ClockCycles(clock, 1)

        assert ~timeout, "Operation timed out!"

    # check if the processors started or stopped a token when expected
    fmt=np.vectorize(lambda x: "{}{}".format(int(x[0]),int(x[1])))
    assert np.all(all_did_startstop == all_should_startstop), "token_start does not match the reference model: {}".format(diff_string(fmt(all_did_startstop), fmt(all_should_startstop), "Step ", "Proc "))
