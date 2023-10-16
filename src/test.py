from typing import Union
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles
import numpy as np
from scipy import sparse as sps
from ttt_pyttt import PyTTT

def bits_to_integers(bit_string: np.ndarray, shape: tuple, dtype = np.int8):
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

    bit_array = np.fromiter(bit_string, dtype=int)
    shape = np.asarray(shape, dtype=int)

    assert np.all(np.isin(bit_array, [0, 1])), "bit array must only contain 0 and 1"
    assert np.all(shape > 0), "shape must only contain positive values"

    bit_depths = len(bit_array) // np.prod(shape)
    value_array = bit_array.reshape((*shape, -1)).dot(1 << np.arange(bit_depths-1,-1,-1))

    if isinstance(dtype, np.dtype):
        return value_array.astype(dtype)
    else:
        return np.array(value_array).astype(dtype)
    
def diff_string(mat1: np.ndarray, mat2: np.ndarray, row_labels: Union[str, list[str]]="row", col_labels: Union[str, list[str]]="col", indent=2):
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
    assert isinstance(row_labels, str) or len(row_labels) == mat1.shape[0], "row_labels must be a string or have length {}".format(mat1.shape[0])
    assert isinstance(col_labels, str) or len(col_labels) == mat1.shape[1], "col_labels must be a string or have length {}".format(mat1.shape[1])

    matrix = sps.csr_matrix(mat1 != mat2)

    if isinstance(row_labels, str):
        row_labels = ["{}{}".format(row_labels, i) for i in range(matrix.shape[0])]
    if isinstance(col_labels, str):
        col_labels = ["{}{}".format(col_labels, i) for i in range(matrix.shape[1])]

    return "\n".join([
        "\n{}{}: {}".format(
            " "*indent,
            row_labels[row], 
            "".join([
                "\n{}  {}: {}!={}".format(" "*indent,col_labels[matrix.indices[j]], mat1[row,matrix.indices[j]], mat2[row,matrix.indices[j]]) 
                for j in range(matrix.indptr[row],matrix.indptr[row+1])
            ])
        ) 
        for row in range(matrix.shape[0]) if matrix.indptr[row] != matrix.indptr[row+1]
    ])

@cocotb.coroutine
async def program(dut, goodTokenThreshold: np.ndarray, badTokenThreshold: np.ndarray, W_good: np.ndarray, W_bad: np.ndarray, duration: np.ndarray):
    NUM_PROCESSORS = len(goodTokenThreshold)
    assert goodTokenThreshold.shape == badTokenThreshold.shape == duration.shape == (NUM_PROCESSORS,), "shape of goodTokenThreshold, badTokenThreshold and duration must be equal to ({},)".format(NUM_PROCESSORS)
    assert W_good.shape == W_bad.shape == (NUM_PROCESSORS, NUM_PROCESSORS), "shape of W_good and W_bad must be equal to ({}, {})".format(NUM_PROCESSORS, NUM_PROCESSORS)

    dut._log.info("programming cores ...")
    dut.reset.value = 1

    for i in range(NUM_PROCESSORS):
        # select the neuron with id i
        dut.neuron_id.value = i

        # set the duration to DURATION
        dut.prog_header.value = 0b001
        dut.prog_data.value = int(duration[i])
        await ClockCycles(dut.clock_fast, 1)

        # set the good token threshold
        dut.prog_header.value = 0b010
        dut.prog_data.value = int(goodTokenThreshold[i])
        await ClockCycles(dut.clock_fast, 1)

        # set the bad token threshold
        dut.prog_header.value = 0b011
        dut.prog_data.value = int(badTokenThreshold[i])
        await ClockCycles(dut.clock_fast, 1)

        if False:
            # start writing the good token weights
            dut.prog_header.value = 0b100
            for j in range(NUM_PROCESSORS):
                dut.prog_data.value = int(W_good[i,j])
                await ClockCycles(dut.clock_fast, 1)

            # start writing the bad token weights
            dut.prog_header.value = 0b101
            for j in range(NUM_PROCESSORS):
                dut.prog_data.value = int(W_bad[i,j])
                await ClockCycles(dut.clock_fast, 1)
    
    dut.neuron_id.value = 0
    dut.prog_header.value = 0b000
    dut.prog_data.value = 0
    dut.reset.value = 0

@cocotb.coroutine
async def inject_tokens(dut, good_tokens: np.ndarray, bad_tokens: np.ndarray):
    NUM_PROCESSORS = len(good_tokens)
    assert good_tokens.shape == bad_tokens.shape == (NUM_PROCESSORS,), "shape of good_tokens and bad_tokens must be equal to ({},)".format(NUM_PROCESSORS)

    for i in np.nonzero(good_tokens)[0]:
        # select the neuron with id i
        dut.neuron_id.value = i

        # inject good tokens
        dut.new_good_tokens.value = int(good_tokens[i])
        await ClockCycles(dut.clock_fast, 1)
        dut.new_good_tokens.value = 0

    for i in np.nonzero(bad_tokens)[0]:
        # select the neuron with id i
        dut.neuron_id.value = i

        # inject bad tokens
        dut.new_bad_tokens.value = int(bad_tokens[i])
        await ClockCycles(dut.clock_fast, 1)
        dut.new_bad_tokens.value = 0

    dut.new_good_tokens.value = 0
    dut.new_bad_tokens.value = 0

@cocotb.test(skip=True)
async def test_core(dut):
    dut = dut.tb_processor_core
    NUM_PROCESSORS = 10
    DURATION = 2
    THRESHOLD = 2

    dut._log.info("start")
    clock_fast = Clock(dut.clock_fast, 20, units="ns")
    cocotb.start_soon(clock_fast.start())

    await ClockCycles(dut.clock_fast, 3)

    goodTokenThreshold = np.zeros((NUM_PROCESSORS,),dtype=int)+THRESHOLD
    badTokenThreshold = np.zeros((NUM_PROCESSORS,),dtype=int)+THRESHOLD
    W_good = np.zeros((NUM_PROCESSORS, NUM_PROCESSORS),dtype=int)
    W_bad = np.zeros((NUM_PROCESSORS, NUM_PROCESSORS),dtype=int)
    duration = np.zeros((NUM_PROCESSORS,),dtype=int)+DURATION

    await program(dut, 
        goodTokenThreshold, 
        badTokenThreshold, 
        W_good,
        W_bad,
        duration
    )

    await ClockCycles(dut.clock_fast, 1)

    _good_tokens_threshold = bits_to_integers(dut.proc.good_tokens_threshold.value, (dut.NUM_PROCESSORS,), dtype=np.uint8)
    _bad_tokens_threshold = bits_to_integers(dut.proc.bad_tokens_threshold.value, (dut.NUM_PROCESSORS,), dtype=np.uint8)
    _duration = bits_to_integers(dut.proc.duration.value, (dut.NUM_PROCESSORS,), dtype=np.uint8)

    # make sure the correct values were programmed in
    assert np.all(_good_tokens_threshold == goodTokenThreshold), "good token threshold not programmed correctly (observed {} != {})".format(_good_tokens_threshold, goodTokenThreshold)
    assert np.all(_bad_tokens_threshold == badTokenThreshold), "bad token threshold not programmed correctly (observed {} != {})".format(_bad_tokens_threshold, badTokenThreshold)
    assert np.all(_duration == duration), "duration not programmed correctly (observed {} != {}) for neuron {}".format(_duration, duration)

    await ClockCycles(dut.clock_fast, 3)

    # trigger the first event by injecting one good token every step for four cycles
    dut.new_good_tokens.value = 1
    await ClockCycles(dut.clock_fast, 4)
    dut.new_good_tokens.value = 0
    await ClockCycles(dut.clock_fast, 4)
    
    # start subtracting tokens
    dut.new_good_tokens.value = -1
    await ClockCycles(dut.clock_fast, 4)
    dut.new_good_tokens.value = 0
    await ClockCycles(dut.clock_fast, 4)

    # add and remove some more token with no effect
    dut.new_good_tokens.value = 1
    await ClockCycles(dut.clock_fast, 2)
    dut.new_good_tokens.value = -1
    await ClockCycles(dut.clock_fast, 2)
    dut.new_good_tokens.value = 0

    await ClockCycles(dut.clock_fast, 10)

    # trigger the second event by injecting one good token every step for four cycles
    dut.new_good_tokens.value = 1
    await ClockCycles(dut.clock_fast, 4)
    dut.new_good_tokens.value = 0

    # now start injecting bad tokens to stop event
    dut.new_bad_tokens.value = 1
    await ClockCycles(dut.clock_fast, 4)
    # now start removing bad tokens to potentially re-trigger event
    dut.new_bad_tokens.value = -1
    await ClockCycles(dut.clock_fast, 4)
    dut.new_bad_tokens.value = 0
    
    # start subtracting good tokens again
    dut.new_good_tokens.value = -1
    await ClockCycles(dut.clock_fast, 4)
    dut.new_good_tokens.value = 0

    await ClockCycles(dut.clock_fast, 15)
    

@cocotb.test()
async def test_core_against_golden_model_without_weights(dut):
    dut = dut.tb_processor_core
    NUM_PROCESSORS = int(dut.NUM_PROCESSORS)
    NUM_SAMPLES = 100
    DELAY = 10

    # set the parameters
    goodTokensThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    badTokensThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    
    W_good = 0*np.random.poisson(1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
    W_bad = 0*np.random.poisson(1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
    duration = np.random.poisson(5, (NUM_PROCESSORS,)).astype(int)

    #OVERWRITE:
    badTokensThreshold[:]=5
    duration[:]=5

    # generate a random number of incoming tokens for each processor
    my_good_tokens_in = np.random.poisson(1, size=(NUM_SAMPLES, NUM_PROCESSORS)).astype(int)
    my_bad_tokens_in = np.random.poisson(0, size=(NUM_SAMPLES, NUM_PROCESSORS)).astype(int)

    # give each incoming token a lifetime of DELAY
    PyTTT.set_expiration(my_good_tokens_in, DELAY)
    PyTTT.set_expiration(my_bad_tokens_in, DELAY)

    # start simulation
    dut._log.info("start")
    clock_fast = Clock(dut.clock_fast, 20, units="ns")
    cocotb.start_soon(clock_fast.start())
    dut.clock_slow.value = 1

    await ClockCycles(dut.clock_fast, 3)

    # create the golden reference model
    golden = PyTTT(goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration)
    # program the same parameters into the hardware
    await program(dut, goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration)

    # run both implementations and compare the results
    all_should_start = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=bool)
    all_should_stop = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=bool)
    all_did_start = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=bool)
    all_did_stop = np.zeros((NUM_SAMPLES, NUM_PROCESSORS), dtype=bool)
    
    # first, record the reference signals
    for (step, (should_start, should_stop)) in enumerate(golden.run(my_good_tokens_in, my_bad_tokens_in)):
        all_should_start[step,:] = should_start
        all_should_stop[step,:] = should_stop

    # then record our hardware implementation
    for (step,(good_in, bad_in)) in enumerate(zip(my_good_tokens_in, my_bad_tokens_in)):
        # do one update-step for the processor for each neuron
        for i in range(NUM_PROCESSORS+1):
            # select the neuron with id i
            dut.neuron_id.value = i

            # inject tokens
            if i < NUM_PROCESSORS:
                dut.new_good_tokens.value = int(good_in[i])
                dut.new_bad_tokens.value = int(bad_in[i])
            else:
                dut.new_good_tokens.value = 0
                dut.new_bad_tokens.value = 0

            await ClockCycles(dut.clock_fast,1)

            await FallingEdge(dut.clock_fast)
            # log the outputs
            if i > 0:
                all_did_start[step,i-1] = bool(dut.token_start.value)
                all_did_stop[step,i-1] = bool(dut.token_stop.value)
            
        dut.new_good_tokens.value = 0
        dut.new_bad_tokens.value = 0

    # check if the processors started or stopped a token when expected
    assert np.all(all_should_start == all_did_start), "token_start does not match the reference model: {}".format(diff_string(all_should_start, all_did_start, "Step", "Processor"))
    assert np.all(all_should_start == all_did_start), "token_stop does not match the reference model: {}".format(diff_string(all_should_stop, all_did_stop, "Step", "Processor"))
