from typing import Union
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles
import numpy as np
from scipy import sparse as sps
from ttt_pyttt import PyTTT

def unpack_bits_to_array(bits: cocotb.binary.BinaryValue, shape: tuple, dtype = np.int8):
    """Converts a bit array to a numpy array of integers.
    
    Parameters
    ----------
    bit_string : array_like
        Bit array to convert.
    shape : tuple
        Shape of the resulting array. The ratio len(bit_string)//prod(shape)  gives the number of bits per value.
    dtype : numpy.dtype or function
        Data type or converter of the resulting array, computed by applying dtype to the bit array.
    """

    shape = np.asarray(shape, dtype=int)

    assert np.all(shape > 0), "shape must only contain positive values"

    bit_depths = len(bits) // np.prod(shape)
    value_array = np.array(
        [bits[(i)*bit_depths:(i+1)*bit_depths-1].integer 
        for i in reversed(range(np.prod(shape)))]
    ).reshape(shape).astype(dtype)

    if isinstance(dtype, np.dtype):
        return value_array.astype(dtype)
    else:
        return np.array(value_array).astype(dtype)
    
def diff_string(mat1: np.ndarray, mat2: np.ndarray, *labels, indent=2):
    """Returns a string representation of the (sparse) difference between two matrices with row and column labels.
    
    Parameters
    ----------
    mat1: array_like
        First matrix.
    mat2: array_like
        Second matrix.
    row_labels : str or array_like
        Row labels. If a string, it will be used as a prefix for the row number.
    col_labels : str or array_like
        Column labels. If a string, it will be used as a prefix for the column number.
    indent : int
        Number of spaces to indent each line.
    """
    assert mat1.shape == mat2.shape, "matrices must have the same shape"
    if len(labels) == 0:
        labels = ["Ax{}".format(i) for i in range(mat1.shape)]

    for i,label in enumerate(labels):
        assert isinstance(label, str) or len(label) == mat1.shape[i], "label(s) for each axis must be a string or have length {}".format(mat1.shape[0])

    nz_indices = np.nonzero(mat1 != mat2)

    # generate the concrete labels for all data points
    concrete_labels = []
    for i,(label,nz) in enumerate(zip(labels, nz_indices)):
        ax_labels = []
        if isinstance(label, str):
            ax_labels = ["{}{}".format(label, idx) for idx in nz]
        else:
            ax_labels = ["{}{}".format(l, idx) for (l,idx) in zip(label, nz)]
        concrete_labels.append(ax_labels)
    #concrete_labels = np.array(concrete_labels)

    fmt=np.vectorize(lambda x: "{}{}".format(int(x[0]),int(x[1])))
    print(fmt(mat1), "\n\n", fmt(mat2))

    return "\n".join([
        "Mismatch at ({}): {} != {}".format(
            ",".join(l),
            v1,
            v2
        ) for (l, v1, v2) in zip(zip(*concrete_labels), mat1[*nz_indices], mat2[*nz_indices])
    ])

@cocotb.coroutine
async def program_processor(clock, dut, goodTokenThreshold: np.ndarray, badTokenThreshold: np.ndarray, duration: np.ndarray):
    NUM_PROCESSORS = len(goodTokenThreshold)
    assert goodTokenThreshold.shape == badTokenThreshold.shape == duration.shape == (NUM_PROCESSORS,), "shape of goodTokenThreshold, badTokenThreshold and duration must be equal to ({},)".format(NUM_PROCESSORS)

    dut._log.info("programming cores ...")

    # push context
    old_reset = dut.reset.value
    old_neuron_id = dut.neuron_id.value
    old_instruction = dut.instruction.value
    old_prog_data = dut.prog_data.value

    dut.reset.value = 0

    for i in range(NUM_PROCESSORS):
        # select the neuron with id i
        dut.neuron_id.value = i

        # set the duration to DURATION
        dut.instruction.value = 0b001
        dut.prog_data.value = int(duration[i])
        await ClockCycles(clock, 1)

        # set the good token threshold
        dut.instruction.value = 0b010
        dut.prog_data.value = int(goodTokenThreshold[i])
        await ClockCycles(clock, 1)

        # set the bad token threshold
        dut.instruction.value = 0b011
        dut.prog_data.value = int(badTokenThreshold[i])
        await ClockCycles(clock, 1)
    
    #pop context
    dut.neuron_id.value = old_neuron_id
    dut.instruction.value = old_instruction
    dut.prog_data.value = old_prog_data
    dut.reset.value = old_reset

@cocotb.coroutine
async def program_network(clock, dut, W_good: np.ndarray, W_bad: np.ndarray):
    NUM_PROCESSORS = int(W_good.shape[0])
    NUM_CONNECTIONS = int(dut.NUM_CONNECTIONS)
    assert W_good.shape == W_bad.shape == (NUM_PROCESSORS, NUM_PROCESSORS), "shape of W_good and W_bad must be equal to ({}, {})".format(NUM_PROCESSORS, NUM_PROCESSORS)

    # get the sparse connectivity
    has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))
    assert has_connection.nnz <= NUM_CONNECTIONS, "network has {} connections, but only {} are available".format(has_connection.nnz, NUM_CONNECTIONS)

    dut._log.info("programming network ...")

    # wait one clock cycle for safety
    await ClockCycles(clock, 1)

    # push context
    old_reset = dut.reset.value
    old_source_id = dut.source_id.value
    old_connection_id = dut.connection_id.value
    old_instruction = dut.instruction.value
    old_prog_data = dut.prog_data.value

    dut.reset.value = 0

    # write all indptrs
    dut.instruction.value = 0b010
    for i in range(NUM_PROCESSORS+1):
        # write indptr for neuron i
        dut.source_id.value = i
        dut.prog_data.value = int(has_connection.indptr[i])
        await ClockCycles(clock, 1)

    # write all data 
    for i,(frm,to) in enumerate(zip(has_connection.indptr[:-1], has_connection.indptr[1:])):
        #write data for neuron i
        dut.source_id.value = i

        for j in range(frm, to):
            dut.connection_id.value = j

            # write index
            dut.instruction.value = 0b011
            dut.prog_data.value = int(has_connection.indices[j])
            await ClockCycles(clock, 1)

            # write good token weight
            dut.instruction.value = 0b100
            dut.prog_data.value = int(W_good[i,has_connection.indices[j]])
            await ClockCycles(clock, 1)

            # write bad token weight
            dut.instruction.value = 0b101
            dut.prog_data.value = int(W_bad[i,has_connection.indices[j]])
            await ClockCycles(clock, 1)

    # pop context
    dut.source_id.value = old_source_id
    dut.connection_id.value = old_connection_id
    dut.instruction.value = old_instruction
    dut.prog_data.value = old_prog_data
    dut.reset.value = old_reset

    # wait one clock cycle for safety
    await ClockCycles(clock, 1)


@cocotb.coroutine
async def inject_tokens(clock, dut, good_tokens: np.ndarray, bad_tokens: np.ndarray):
    NUM_PROCESSORS = len(good_tokens)
    assert good_tokens.shape == bad_tokens.shape == (NUM_PROCESSORS,), "shape of good_tokens and bad_tokens must be equal to ({},)".format(NUM_PROCESSORS)

    # push context
    old_neuron_id = dut.neuron_id.value
    old_instruction = dut.instructions.value
    old_new_good_tokens = dut.new_good_tokens.value
    old_new_bad_tokens = dut.new_bad_tokens.value

    # set operation to run
    dut.instructions.value = 0b100

    for i in np.nonzero(good_tokens)[0]:
        # select the neuron with id i
        dut.neuron_id.value = i

        # inject good tokens
        dut.new_good_tokens.value = int(good_tokens[i])
        await ClockCycles(clock, 1)

    for i in np.nonzero(bad_tokens)[0]:
        # select the neuron with id i
        dut.neuron_id.value = i

        # inject bad tokens
        dut.new_bad_tokens.value = int(bad_tokens[i])
        await ClockCycles(clock, 1)

    # pop context
    dut.neuron_id.value = old_neuron_id
    dut.instructions.value = old_instruction
    dut.new_good_tokens.value = old_new_good_tokens
    dut.new_bad_tokens.value = old_new_bad_tokens

@cocotb.test()
async def test_core_programming(dut):
    dut = dut.tb_processor_core
    NUM_PROCESSORS = 10
    DURATION = 2
    THRESHOLD = 2

    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clock_fast, 20, units="ns").start())
    clock = dut.clock_fast

    await ClockCycles(clock, 3)

    goodTokenThreshold = np.zeros((NUM_PROCESSORS,),dtype=int)+THRESHOLD
    badTokenThreshold = np.zeros((NUM_PROCESSORS,),dtype=int)+THRESHOLD
    duration = np.zeros((NUM_PROCESSORS,),dtype=int)+DURATION

    await program_processor(clock,
        dut, 
        goodTokenThreshold, 
        badTokenThreshold,
        duration
    )

    await ClockCycles(clock, 1)


    _good_tokens_threshold = unpack_bits_to_array(dut.proc.good_tokens_threshold.value, (dut.NUM_PROCESSORS,), dtype=np.uint8)
    _bad_tokens_threshold = unpack_bits_to_array(dut.proc.bad_tokens_threshold.value, (dut.NUM_PROCESSORS,), dtype=np.uint8)
    _duration = unpack_bits_to_array(dut.proc.duration.value, (dut.NUM_PROCESSORS,), dtype=np.uint8)

    # make sure the correct values were programmed in
    assert np.all(_good_tokens_threshold == goodTokenThreshold), "good token threshold not programmed correctly (observed {} != {})".format(_good_tokens_threshold, goodTokenThreshold)
    assert np.all(_bad_tokens_threshold == badTokenThreshold), "bad token threshold not programmed correctly (observed {} != {})".format(_bad_tokens_threshold, badTokenThreshold)
    assert np.all(_duration == duration), "duration not programmed correctly (observed {} != {}) for neuron {}".format(_duration, duration)

    return
    # IGNORE THE REST FOR NOW
    await ClockCycles(clock, 3)

    # trigger the first event by injecting one good token every step for four cycles
    dut.new_good_tokens.value = 1
    await ClockCycles(clock, 4)
    dut.new_good_tokens.value = 0
    await ClockCycles(clock, 4)
    
    # start subtracting tokens
    dut.new_good_tokens.value = -1
    await ClockCycles(clock, 4)
    dut.new_good_tokens.value = 0
    await ClockCycles(clock, 4)

    # add and remove some more token with no effect
    dut.new_good_tokens.value = 1
    await ClockCycles(clock, 2)
    dut.new_good_tokens.value = -1
    await ClockCycles(clock, 2)
    dut.new_good_tokens.value = 0

    await ClockCycles(clock, 10)

    # trigger the second event by injecting one good token every step for four cycles
    dut.new_good_tokens.value = 1
    await ClockCycles(clock, 4)
    dut.new_good_tokens.value = 0

    # now start injecting bad tokens to stop event
    dut.new_bad_tokens.value = 1
    await ClockCycles(clock, 4)
    # now start removing bad tokens to potentially re-trigger event
    dut.new_bad_tokens.value = -1
    await ClockCycles(clock, 4)
    dut.new_bad_tokens.value = 0
    
    # start subtracting good tokens again
    dut.new_good_tokens.value = -1
    await ClockCycles(clock, 4)
    dut.new_good_tokens.value = 0

    await ClockCycles(clock, 15)
    

@cocotb.test()
async def test_core_against_golden_model_without_weights(dut):
    dut = dut.tb_processor_core
    NUM_PROCESSORS = int(dut.NUM_PROCESSORS)
    NUM_SAMPLES = 100
    DELAY = 10

    # set the parameters
    goodTokensThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    badTokensThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    duration = np.random.poisson(5, (NUM_PROCESSORS,)).astype(int)

    #OVERWRITE:
    badTokensThreshold[:]=5
    duration[:]=5

    # generate a random number of incoming tokens for each processor
    my_good_tokens_in = np.random.poisson(1, size=(NUM_SAMPLES, NUM_PROCESSORS)).astype(int)
    my_bad_tokens_in = np.random.poisson(0, size=(NUM_SAMPLES, NUM_PROCESSORS)).astype(int)
    

    # make sure the first processor fires in the first timestep, because this was a bug before
    goodTokensThreshold[0]=0
    my_good_tokens_in[0,0] = goodTokensThreshold[0]+1
    my_bad_tokens_in[0,0] = 0

    # give each incoming token a lifetime of DELAY
    PyTTT.set_expiration(my_good_tokens_in, DELAY)
    PyTTT.set_expiration(my_bad_tokens_in, DELAY)

    # start simulation
    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clock_fast, 20, units="ns").start())
    clock = dut.clock_fast

    dut.clock_slow.value = 1
    dut.neuron_id.value = 0

    await ClockCycles(clock, 3)
    dut.reset.value = 1
    dut.instruction.value = 0b000

    # reset every neuron
    for i in range(NUM_PROCESSORS):
        # select the neuron with id i and wait one cycle to reset
        dut.neuron_id.value = i
        await ClockCycles(clock, 1)

    dut.reset.value = 0
    await ClockCycles(clock, 1)

    # create the golden reference model
    golden = PyTTT(goodTokensThreshold, badTokensThreshold, np.zeros((NUM_PROCESSORS,NUM_PROCESSORS),dtype=int), np.zeros((NUM_PROCESSORS,NUM_PROCESSORS),dtype=int), duration)
    # program the same parameters into the hardware
    await program_processor(clock, dut, goodTokensThreshold, badTokensThreshold, duration)

    # run both implementations and compare the results
    all_should_startstop = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=object)
    all_did_startstop = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=object)
    
    # first, record the reference signals
    for (step, (should_start, should_stop)) in enumerate(golden.run(my_good_tokens_in, my_bad_tokens_in)):
        all_should_startstop[step,:] = list(zip(should_start, should_stop))

    # then record our hardware implementation
    for (step,(good_in, bad_in)) in enumerate(zip(my_good_tokens_in, my_bad_tokens_in)):
        # do one update-step for the processor for each neuron to count incoming tokens
        dut.instruction.value = 0b100
        for i in range(NUM_PROCESSORS):
            # select the neuron with id i
            dut.neuron_id.value = i

            # inject tokens
            dut.new_good_tokens.value = int(good_in[i])
            dut.new_bad_tokens.value = int(bad_in[i])

            await ClockCycles(clock, 1)
            

        # do one update-step for the processor for each neuron to update internal states and record the outputs
        dut.instruction.value = 0b101
        for i in range(NUM_PROCESSORS):
            # select the neuron with id i
            dut.neuron_id.value = i
            await RisingEdge(clock)
            await FallingEdge(clock)
            # log the outputs
            all_did_startstop[step,i] = tuple(map(bool, dut.token_startstop.value))
            
        dut.new_good_tokens.value = 0
        dut.new_bad_tokens.value = 0

    # check if the processors started or stopped a token when expected
    assert np.all(all_did_startstop == all_should_startstop), "token_start does not match the reference model: {}".format(diff_string(all_did_startstop, all_should_startstop, "Step ", "Proc "))

@cocotb.test()
async def test_network_programming(dut):
    dut = dut.tb_network
    NUM_PROCESSORS = int(dut.NUM_PROCESSORS)
    NUM_CONNECTIONS = int(dut.NUM_CONNECTIONS)
    NEW_TOKENS_BITS = int(dut.NEW_TOKENS_BITS)
    NUM_SAMPLES = 100

    # start simulation
    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk

    # reset
    dut.reset.value = 1
    await ClockCycles(clock, 3)
    dut.reset.value = 0
    await ClockCycles(clock, 3)

    # program in the weights
    while True:
        W_good = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        W_bad = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        
        has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))

        if has_connection.nnz <= NUM_CONNECTIONS:
            break

    await ClockCycles(clock, 3)

    await program_network(clock, dut, W_good, W_bad)

    # ceck if the weights were programmed in correctly
    _indptr = np.array([x.integer for x in reversed(dut.net.tgt_indptr.value)]).astype(np.uint)
    _indices = np.array([x.integer for x in reversed(dut.net.tgt_indices.value)])[:_indptr[-1]].astype(np.uint)
    _new_good_tokens = np.array([x.integer for x in reversed(dut.net.tgt_new_good_tokens.value)])[:_indptr[-1]].astype(int)
    _new_bad_tokens = np.array([x.integer for x in reversed(dut.net.tgt_new_bad_tokens.value)])[:_indptr[-1]].astype(int)

    row_idx,col_idx = has_connection.nonzero()

    # make sure the correct values were programmed in
    assert len(_indices) == has_connection.nnz, "number of connections does not match (observed {} != {})".format(len(_indices), has_connection.nnz)
    assert np.all(_indptr == has_connection.indptr), "indptr not programmed correctly (observed {} != {})".format(_indptr, has_connection.indptr)
    assert np.all(_indices == has_connection.indices), "indices not programmed correctly (observed {} != {})".format(_indices, has_connection.indices)
    assert np.all(_new_good_tokens == [W_good[r,c] for r,c in zip(row_idx,col_idx)]), "good token weights not programmed correctly (observed {} != {})".format(_new_good_tokens, [W_good[r,c] for r,c in zip(row_idx,col_idx)])
    assert np.all(_new_bad_tokens == [W_bad[r,c] for r,c in zip(row_idx,col_idx)]), "bad token weights not programmed correctly (observed {} != {})".format(_new_bad_tokens, [W_bad[r,c] for r,c in zip(row_idx,col_idx)])


@cocotb.test()
async def test_network_cycle(dut):
    dut = dut.tb_network
    NUM_PROCESSORS = int(dut.NUM_PROCESSORS)
    NUM_CONNECTIONS = int(dut.NUM_CONNECTIONS)
    NEW_TOKENS_BITS = int(dut.NEW_TOKENS_BITS)
    NUM_SAMPLES = 100

    dut._log.info("start")
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    clock = dut.clk

    # reset
    dut.reset.value = 1
    await ClockCycles(clock, 3)
    dut.reset.value = 0
    await ClockCycles(clock, 3)

    # program in the weights
    while True:
        W_good = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        W_bad = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))
        if has_connection.nnz <= NUM_CONNECTIONS:
            break

    await ClockCycles(clock, 3)
    await program_network(clock, dut, W_good, W_bad)
    await ClockCycles(clock, 3)

    # ceck if the weights were programmed in correctly
    _indptr = np.array([x.integer for x in reversed(dut.net.tgt_indptr.value)]).astype(np.uint)
    _indices = np.array([x.integer for x in reversed(dut.net.tgt_indices.value)])[:_indptr[-1]].astype(np.uint)

    did_done = []
    should_done = []
    did_tgt_addr = []
    should_tgt_addr = []
    did_end_addr = []
    should_end_addr = []
    did_index = []
    should_index = []
    did_good = []
    should_good = []
    did_bad = []
    should_bad = []

    # cycle over the memory
    for i in range(NUM_PROCESSORS):
        dut.source_id.value = i
        # start iterating
        dut.instruction.value = 0b110
        await RisingEdge(clock)
        await FallingEdge(clock)
        
        # record given and correct answers
        did_done.append(dut.done.value.integer)
        should_done.append(0)
        did_tgt_addr.append(dut.net.tgt_addr.value.integer)
        should_tgt_addr.append(_indptr[i])
        did_end_addr.append(dut.net.end_addr.value.integer)
        should_end_addr.append(_indptr[i+1])

        # start cycling
        dut.instruction.value = 0b111
        # now the results should start coming
        for j in range(_indptr[i], _indptr[i+1]):
            await RisingEdge(clock)
            await FallingEdge(clock)
            idx = dut.target_id.value.integer

            # record given and correct answers
            did_index.append(idx)
            should_index.append(_indices[j])
            did_good.append(dut.new_good_tokens.value.integer)
            should_good.append(W_good[i,idx])
            did_bad.append(dut.new_bad_tokens.value.integer)
            should_bad.append(W_bad[i,idx])
            
        await FallingEdge(clock)
        did_done.append(dut.done.value.integer)
        should_done.append(1)

    # check all the outputs

    did_done = np.array(did_done)
    should_done = np.array(should_done)
    did_tgt_addr = np.array(did_tgt_addr)
    should_tgt_addr = np.array(should_tgt_addr)
    did_end_addr = np.array(did_end_addr)
    should_end_addr = np.array(should_end_addr)
    did_index = np.array(did_index)
    should_index = np.array(should_index)
    did_good = np.array(did_good)
    should_good = np.array(should_good)
    did_bad = np.array(did_bad)
    should_bad = np.array(should_bad)
    assert np.all(did_done == should_done), "done signal does not match the reference model: {}".format(diff_string(did_done, should_done, "Step"))
    assert np.all(did_tgt_addr == should_tgt_addr), "tgt_addr signal does not match the reference model: {}".format(diff_string(did_tgt_addr, should_tgt_addr, "Step"))
    assert np.all(did_end_addr == should_end_addr), "tgt_end signal does not match the reference model: {}".format(diff_string(did_end_addr, should_end_addr, "Step"))
    assert np.all(did_index == should_index), "index signal does not match the reference model: {}".format(diff_string(did_index, should_index, "Step"))
    assert np.all(did_good == should_good), "good signal does not match the reference model: {}".format(diff_string(did_good, should_good, "Step"))
    assert np.all(did_bad == should_bad), "bad signal does not match the reference model: {}".format(diff_string(did_bad, should_bad, "Step"))

