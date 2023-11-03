from typing import Union, Optional
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles
from cocotb.utils import get_sim_time
import numpy as np
from scipy import sparse as sps
from ttt_pyttt import PyTTT
from utils import *
from collections import namedtuple

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
    #dut._log.info("programming core...")

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
        assert dut.uo_out.value.integer == to_n_bit_twos_complement(goodTokenCount,8), "good token count not programmed correctly (observed {} != {})".format(dut.uo_out.value, goodTokenCount)

    if badTokenCount is not None:
        # set the bad token count
        dut.ui_in.value  = 0b0100 | (flags << 4)
        dut.uio_in.value = 0b11111111 & int(badTokenCount)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == to_n_bit_twos_complement(badTokenCount,8), "bad token count not programmed correctly (observed {} != {})".format(dut.uo_out.value, badTokenCount)

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
    
    #dut._log.info("getting processor state...")

    assert dut.rst_n.value == 1, "Cannot read processor state during reset!"

    observed = {}
    await RisingEdge(clock)
    await FallingEdge(clock)

    if duration is not False:
        # get the duration
        dut.ui_in.value  = 0b1111
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["duration"] = dut.uo_out.value.integer

    if goodTokenThreshold is not False:
        # get the good token threshold
        dut.ui_in.value  = 0b1011
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["goodTokenThreshold"] = dut.uo_out.value.integer

    if badTokenThreshold is not False:
        # get the bad token threshold
        dut.ui_in.value  = 0b1101
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["badTokenThreshold"] = dut.uo_out.value.integer
    
    if goodTokenCount is not False:
        # get the good token count
        dut.ui_in.value  = 0b0011
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["goodTokenCount"] = dut.uo_out.value.integer

    if badTokenCount is not False:
        # get the bad token count
        dut.ui_in.value  = 0b0101
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["badTokenCount"] = dut.uo_out.value.integer

    if remainingDuration is not False:
        # get the remaining duration
        dut.ui_in.value  = 0b0111
        dut.uio_in.value = 0b00000000
        await RisingEdge(clock)
        await FallingEdge(clock)
        observed["remainingDuration"] = dut.uo_out.value.integer

    # Clean up again
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(clock, 1)
    return observed

def compare_processor_state(observed, expected):
    assert observed["duration"] == expected["duration"], "duration not programmed correctly (observed {} != {})".format(observed["duration"], expected["duration"])
    assert observed["goodTokenThreshold"] == expected["goodTokenThreshold"], "good token threshold not programmed correctly (observed {} != {})".format(observed["goodTokenThreshold"], expected["goodTokenThreshold"])
    assert observed["badTokenThreshold"] == expected["badTokenThreshold"], "bad token threshold not programmed correctly (observed {} != {})".format(observed["badTokenThreshold"], expected["badTokenThreshold"])
    assert observed["goodTokenCount"] == to_n_bit_twos_complement(expected["goodTokenCount"],8), "good token count not programmed correctly (observed {} != {})".format(observed["goodTokenCount"], to_n_bit_twos_complement(expected["goodTokenCount"],8))
    assert observed["badTokenCount"] == to_n_bit_twos_complement(expected["badTokenCount"],8), "bad token count not programmed correctly (observed {} != {})".format(observed["badTokenCount"], to_n_bit_twos_complement(expected["badTokenCount"],8))
    assert observed["remainingDuration"] == expected["remainingDuration"], "remaining duration not programmed correctly (observed {} != {})".format(observed["remainingDuration"], expected["remainingDuration"])


async def stimulate(clock, dut, goodTokens=0, badTokens=0):
    if goodTokens != 0:
        # stimulate with good tokens
        dut.ui_in.value = 0b0000
        dut.uio_in.value = int(goodTokens)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == to_n_bit_twos_complement(goodTokens,8), "stimulation failed; read back wrong value {} != {}".format(dut.uo_out.value.integer, goodTokens)

    if badTokens != 0:
        # stimulate with bad tokens
        dut.ui_in.value = 0b0001
        dut.uio_in.value = int(badTokens)
        await RisingEdge(clock)
        await FallingEdge(clock)
        assert dut.uo_out.value.integer == to_n_bit_twos_complement(badTokens,8), "stimulation failed; read back wrong value {} != {}".format(dut.uo_out.value.integer, badTokens)

    dut.ui_in.value = 0b0000
    dut.uio_in.value = 0b0000

StartStop = namedtuple("StartStop", ["start", "stop"])

async def tally(clock, dut):
    dut.ui_in.value = 0b1000
    dut.uio_in.value = 0b00000000
    await RisingEdge(clock)
    await FallingEdge(clock)
    dut.ui_in.value = 0b0000
    return StartStop(start=dut.uo_out[1].value, stop=dut.uo_out[0].value)

async def countdown(clock, dut):
    dut.ui_in.value = 0b1001
    dut.uio_in.value = 0b00000000
    await RisingEdge(clock)
    await FallingEdge(clock)
    dut.ui_in.value = 0b0000


@cocotb.test()
async def test_programming(dut):
    NUM_TRIALS = 100

    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk
    dut.ena.value = 1

    await ClockCycles(clock, 3)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await ClockCycles(clock, 3)

    for i in range(NUM_TRIALS):
        # generate parameters
        goodTokenThreshold = int(np.random.poisson(1))
        badTokenThreshold = int(np.random.poisson(0.25))
        duration = int(np.random.poisson(5))

        # generate initial conditions
        goodTokenCount = int(np.random.poisson(1))
        badTokenCount = int(np.random.poisson(0.25))
        remainingDuration = int(np.random.poisson(5))

        expected = dict(
            goodTokenThreshold=goodTokenThreshold, 
            badTokenThreshold=badTokenThreshold, 
            duration=duration,
            goodTokenCount=goodTokenCount,
            badTokenCount=badTokenCount,
            remainingDuration=remainingDuration
        )

        # program the simulated hardware
        await program_processor(
            dut.clk, 
            dut, 
            **expected
        )

        await ClockCycles(clock, 3)

        # get processor state
        observed = await get_processor_state(dut.clk, dut )

        compare_processor_state(observed, expected)


@cocotb.test()
async def test_execution(dut):
    NUM_PROCESSORS = 10
    NUM_SAMPLES = 100
    NUM_CONNECTIONS = 50

    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk
    dut.ena.value = 1

    dut.ui_in.value = 0
    dut.uio_in.value = 0

    await ClockCycles(clock, 3)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await RisingEdge(clock)
    await FallingEdge(clock)


    # generate the weights
    while True:
        W_good = np.random.poisson(0.5*NUM_CONNECTIONS/NUM_PROCESSORS**2, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        W_bad = np.random.poisson(0.5*NUM_CONNECTIONS/NUM_PROCESSORS**2, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        
        # make sure the first processor has outgoing connections
        W_good[1 % NUM_PROCESSORS,0] = 1
        W_bad[2 % NUM_PROCESSORS,0] = 1

        has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))

        if has_connection.nnz <= NUM_CONNECTIONS and has_connection.nnz > 0 and W_good.max() < 5 and W_bad.max() < 5:
            break

    #W_good[:,:]=0
    #W_bad[:,:]=0

    # generate parameters
    goodTokenThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    badTokenThreshold = np.random.poisson(0.25, (NUM_PROCESSORS,)).astype(int)
    duration = np.random.poisson(5, (NUM_PROCESSORS,)).astype(int)

    initialGoodTokens = np.zeros((NUM_PROCESSORS,), dtype=int)
    initialBadTokens = np.zeros((NUM_PROCESSORS,), dtype=int)
    initialDuration = np.zeros((NUM_PROCESSORS,), dtype=int)

    # make sure that processor 0, which is written first, is excitable
    goodTokenThreshold[0] = 1
    badTokenThreshold[0] = 1
    duration[0] = 2

    # create a golden reference model
    golden = PyTTT(
        goodTokenThreshold, 
        badTokenThreshold, 
        W_good,
        W_bad, 
        duration, 
        initialGoodTokens, 
        initialBadTokens, 
        initialDuration
    )

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
        for proc,(st,sp) in enumerate(zip(should_start, should_stop)):
            all_should_startstop[step,proc] = StartStop(start=st,stop=sp)

    # make sure the reference model works
    assert all_should_startstop[0,0] == StartStop(start=True,stop=False), "first processor in reference model did not start in first step, but we programmed it to do so!"

    start_token = np.zeros((NUM_PROCESSORS,), dtype=bool)
    stop_token = np.zeros((NUM_PROCESSORS,), dtype=bool)
    start_token_prev = np.zeros_like(start_token)
    stop_token_prev = np.zeros_like(stop_token)
    states = [{
        "goodTokenThreshold": goodTokenThreshold[i],
        "badTokenThreshold": badTokenThreshold[i],
        "duration": duration[i],
        "goodTokenCount": initialGoodTokens[i],
        "badTokenCount": initialBadTokens[i],
        "remainingDuration": initialDuration[i],
    } for i in range(NUM_PROCESSORS)]

    # then record our hardware implementation
    for (step,(good_in, bad_in)) in enumerate(zip(my_good_tokens_in, my_bad_tokens_in)):
        #print("Step {}".format(step))

        # update all processors one at a time
        for p in range(NUM_PROCESSORS):
            # load all the processor parameters from memory
            await program_processor( dut.clk, dut, **states[p])

            # add/remove the external tokens to this processor
            await stimulate( dut.clk, dut, good_in[p], bad_in[p] )

            # add the just started recurrent tokens to this processor one by one
            for src,start in enumerate(start_token_prev):
                if start:
                    if step==0:
                        raise AssertionError("Shouldn't have any recurrent inputs in step 0, but started {} good and {} bad tokens from {} to {}".format(W_good[p,src], W_bad[p,src], src,p))
                    await stimulate( dut.clk, dut, W_good[p,src], W_bad[p,src] )

            # remove the just stopped recurrent tokens to this processor one by one
            for src,stop in enumerate(stop_token_prev):
                if stop:
                    if step == 0:
                        raise AssertionError("Shouldn't have any recurrent inputs in step 0, but stopped {} good and {} bad tokens from {} to {}".format(W_good[p,src], W_bad[p,src], src,p))
                    await stimulate( dut.clk, dut, -W_good[p,src], -W_bad[p,src] )

            # for now, just advance at every iteration
            if(True):
                await countdown( dut.clk, dut )

            # finally tally up the sums in the processor and make a decision whether or not to start/stop firing
            startstop = await tally(dut.clk, dut)

            dut.ui_in.value = 0
            
            # record the start/stop signals
            start_token[p] = startstop.start
            stop_token[p] = startstop.stop
            all_did_startstop[step,p] = startstop
            times[step] = get_sim_time("ns")

            # record the processor's internal state (but not the parameters)
            new_state = await get_processor_state( 
                dut.clk, 
                dut, 
                goodTokenThreshold=False, 
                badTokenThreshold=False, 
                duration=False
            )

            # update our internal record
            states[p].update(new_state)
        
            await RisingEdge(clock)
            await FallingEdge(clock)

        # update the start/stop tokens for the next iteration
        start_token_prev[:] = start_token
        stop_token_prev[:] = stop_token

    # check if the processors started or stopped a token when expected
    fmt=np.vectorize(lambda x: "start: {}, stop: {}".format(int(x.start),int(x.stop)))
    assert np.all(all_did_startstop == all_should_startstop), "token_start does not match the reference model: {}".format(diff_string(fmt(all_did_startstop), fmt(all_should_startstop), "Step ", "Proc "))


@cocotb.test()
async def test_token_counters(dut):
    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk
    dut.ena.value = 1

    dut.ui_in.value = 0
    dut.uio_in.value = 0

    await ClockCycles(clock, 1)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await ClockCycles(clock, 1)

    # set good and bad token thresholds to 5
    await program_processor( clock, dut, goodTokenThreshold=5, badTokenThreshold=5 )

    await ClockCycles(clock, 1)

    # send first good and bad token
    await stimulate( clock, dut, goodTokens=1, badTokens=1 )
    
    # do nothing for a while
    await ClockCycles(clock, 3)

    # update the state -> 1,1 -> no token yet
    token_start, token_stop = await tally( clock, dut )
    assert token_start == token_stop == 0, "no tokens should have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)

    # add two more tokens, one-by-one
    await stimulate( clock, dut, goodTokens=1, badTokens=1 )
    await stimulate( clock, dut, goodTokens=1, badTokens=1 )

    # update the state -> 3,3 -> no token yet
    token_start, token_stop = await tally( clock, dut )
    assert token_start == token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)

    # update again -> should have no effect
    token_start, token_stop = await tally( clock, dut )
    assert token_start == token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)

    # add three more tokens in one go
    await stimulate( clock, dut, goodTokens=3, badTokens=3 )
    # update the state -> 6,6 -> should not start a token, because bad tokens block it 
    token_start, token_stop = await tally( clock, dut )
    assert token_start == token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)

    # remove 1 token each
    await stimulate( clock, dut, goodTokens=-1, badTokens=-1 )
    
    # update the state -> 5,5 -> should start a token now
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 1 and token_stop == 0, "the token should have been started, but got start: {}, stop: {}".format(token_start, token_stop)

    await ClockCycles(clock, 5)

    # add one bad token
    await stimulate( clock, dut, badTokens=1 )

    # update the state -> 5,6 -> should stop the token
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 1, "the token should have been stopped, but got start: {}, stop: {}".format(token_start, token_stop)

    # remove two tokens each
    await stimulate( clock, dut, goodTokens=-2, badTokens=-2 )

    # update the state -> 3,4 -> should not start a token
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)

    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=False )
    assert state["goodTokenCount"] == 3 and state["badTokenCount"] == 4, "the token count should be 3,4 but got good: {}, bad: {}".format(state["goodTokenCount"], state["badTokenCount"])


@cocotb.test()
async def test_token_duration(dut):
    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk
    dut.ena.value = 1

    dut.ui_in.value = 0
    dut.uio_in.value = 0

    await ClockCycles(clock, 1)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await ClockCycles(clock, 1)

    # set good and bad token thresholds to 5 and duration to 5
    await program_processor( clock, dut, goodTokenThreshold=5, badTokenThreshold=5, duration=5 )

    await ClockCycles(clock, 1)

    # send 5 good tokens to trigger
    await stimulate( clock, dut, goodTokens=5 )
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 1 and token_stop == 0, "the token should have been started, but got start: {}, stop: {}".format(token_start, token_stop)

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 4, "the remaining duration should be 4, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 3, "the remaining duration should be 3, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 2, "the remaining duration should be 2, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 1, "the remaining duration should be 1, but got {}".format(state["remainingDuration"])


    # count down the duration -> now we should have hit 0, but there is still enough input to trigger a new token,
    # so the token should continue, i.e. not be stopped
    await countdown( clock, dut )

    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 5, "the remaining duration should be 5, but got {}".format(state["remainingDuration"])

    # now take away the input
    await stimulate( clock, dut, goodTokens=-5 )

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 4, "the remaining duration should be 4, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 3, "the remaining duration should be 3, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 2, "the remaining duration should be 2, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 0, "the token should not have been started/stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 1, "the remaining duration should be 1, but got {}".format(state["remainingDuration"])

    # count down the duration
    await countdown( clock, dut )


    # check if the token is still running
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 1, "the token should have been stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 0, "the remaining duration should be 0, but got {}".format(state["remainingDuration"])

    # trigger another token
    await stimulate( clock, dut, goodTokens=5 )
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 1 and token_stop == 0, "the token should have been started, but got start: {}, stop: {}".format(token_start, token_stop)

    # count down the duration
    await countdown( clock, dut )
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 4, "the remaining duration should be 4, but got {}".format(state["remainingDuration"])

    # cancel the token
    await stimulate( clock, dut, goodTokens=-5, badTokens=6 )
    token_start, token_stop = await tally( clock, dut )
    assert token_start == 0 and token_stop == 1, "the token should have been stopped, but got start: {}, stop: {}".format(token_start, token_stop)
    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=True )
    assert state["remainingDuration"] == 0, "the remaining duration should be 0, but got {}".format(state["remainingDuration"])


@cocotb.test()
async def test_write_fast(dut):
    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk
    dut.ena.value = 1

    dut.ui_in.value = 0
    dut.uio_in.value = 0

    await ClockCycles(clock, 1)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await ClockCycles(clock, 1)

    # set good and bad token thresholds to 5 and duration to 5
    await program_processor( clock, dut, goodTokenThreshold=5, badTokenThreshold=5, duration=5 )

    await ClockCycles(clock, 1)
    dut.ui_in.value = 0b0000
    dut.uio_in.value = 1
    await ClockCycles(clock, 5)
    dut.ui_in.value = 0b0001
    dut.uio_in.value = 1
    await ClockCycles(clock, 3)
    dut.ui_in.value = 0b0000
    dut.uio_in.value = 0

    await ClockCycles(clock, 1)

    state = await get_processor_state( clock, dut, goodTokenThreshold=False, badTokenThreshold=False, duration=False, remainingDuration=False )

    assert state["goodTokenCount"] == 5 and state["badTokenCount"] == 3, "the token count should be 5,3 but got good: {}, bad: {}".format(state["goodTokenCount"], state["badTokenCount"])


@cocotb.test()
async def test_reset(dut):
    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk
    dut.ena.value = 1

    dut.ui_in.value = 0
    dut.uio_in.value = 0

    await ClockCycles(clock, 1)
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await ClockCycles(clock, 1)

    # set good and bad token thresholds to 5 and duration to 5
    setState = dict(
        goodTokenThreshold=5, 
        badTokenThreshold=5, 
        duration=5,
        goodTokenCount=5,
        badTokenCount=3,
        remainingDuration=3
    )
    await program_processor( clock, dut, **setState )

    # check state
    state = await get_processor_state( clock, dut )
    compare_processor_state(state, setState)

    # reset
    dut.rst_n.value = 0
    await ClockCycles(clock, 1)
    dut.rst_n.value = 1
    await ClockCycles(clock, 1)

    # check state
    setState["goodTokenCount"] = 0
    setState["badTokenCount"] = 0
    setState["remainingDuration"] = 0
    state = await get_processor_state( clock, dut )
    compare_processor_state(state, setState)

