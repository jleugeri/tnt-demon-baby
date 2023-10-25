from typing import Union
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles
from cocotb.utils import get_sim_time
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

    nz = np.nonzero(mat1 != mat2)
    nz_indices = nz

    # generate the concrete labels for all data points
    concrete_labels = []
    concrete_idx = []
    for i,(label,nz) in enumerate(zip(labels, nz_indices)):
        ax_labels = []
        ax_nz = []
        if isinstance(label, str):
            ax_labels = ["{}{}".format(label, idx) for idx in nz]
        else:
            ax_labels = ["{}{}".format(l, idx) for (l,idx) in zip(label, nz)]
        concrete_labels.append(ax_labels)
        concrete_idx.append(ax_nz)
    #concrete_labels = np.array(concrete_labels)

    return "\n".join([
        "Mismatch at ({}): {} != {}".format(
            ",".join(l),
            mat1[coords],
            mat2[coords]
        ) for (l, coords) in zip(zip(*concrete_labels), zip(*nz_indices))
    ])

async def program_processor(clock, dut, goodTokenThreshold: np.ndarray, badTokenThreshold: np.ndarray, duration: np.ndarray, prefix = 1):
    NUM_PROCESSORS = len(goodTokenThreshold)
    assert goodTokenThreshold.shape == badTokenThreshold.shape == duration.shape == (NUM_PROCESSORS,), "shape of goodTokenThreshold, badTokenThreshold and duration must be equal to ({},)".format(NUM_PROCESSORS)

    dut._log.info("programming cores ...")

    assert dut.reset.value == 0, "reset must be low before programming"

    for i in range(NUM_PROCESSORS):
        # select the neuron with id i
        dut.processor_id.value = i

        # set the duration to DURATION
        dut.instruction.value = 0b01 | (prefix << 2)
        dut.prog_duration.value = int(duration[i])
        await ClockCycles(clock, 1)

        # set the good token threshold
        dut.instruction.value = 0b10 | (prefix << 2)
        dut.prog_threshold.value = int(goodTokenThreshold[i])
        await ClockCycles(clock, 1)

        # set the bad token threshold
        dut.instruction.value = 0b11 | (prefix << 2)
        dut.prog_threshold.value = int(badTokenThreshold[i])
        await ClockCycles(clock, 1)
    
    dut.instruction.value = 0b000
    dut.prog_threshold.value = 0
    dut.prog_duration.value = 0
    dut.processor_id.value = 0
    await ClockCycles(clock, 1)

async def program_network(clock, dut, W_good: np.ndarray, W_bad: np.ndarray, prefix = 1):
    NUM_PROCESSORS = int(W_good.shape[0])
    NUM_CONNECTIONS = int(dut.NUM_CONNECTIONS)
    assert W_good.shape == W_bad.shape == (NUM_PROCESSORS, NUM_PROCESSORS), "shape of W_good and W_bad must be equal to ({}, {})".format(NUM_PROCESSORS, NUM_PROCESSORS)

    assert dut.reset.value == 0, "reset must be low before programming"

    # Transpose W_good and W_bad, because the hardware expects the rows to be the source and the columns to be the target
    W_good = W_good.T
    W_bad = W_bad.T

    # get the sparse connectivity
    has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))
    assert has_connection.nnz <= NUM_CONNECTIONS, "network has {} connections, but only {} are available".format(has_connection.nnz, NUM_CONNECTIONS)

    dut._log.info("programming network ...")

    # wait one clock cycle for safety
    await ClockCycles(clock, 1)

    # write all indptrs
    dut.instruction.value = 0b10 | (prefix << 2)
    for i in range(NUM_PROCESSORS+1):
        # write indptr for neuron i
        dut.processor_id.value = i
        dut.connection_id.value = int(has_connection.indptr[i])
        await ClockCycles(clock, 1)

    # write all data 
    for i,(frm,to) in enumerate(zip(has_connection.indptr[:-1], has_connection.indptr[1:])):
        #write data for neuron i
        dut.processor_id.value = i

        for j in range(frm, to):
            dut.connection_id.value = j

            # write index
            dut.instruction.value = 0b11 | (prefix << 2)
            dut.processor_id.value = int(has_connection.indices[j])
            await ClockCycles(clock, 1)

            # write good token weight
            dut.instruction.value = 0b00 | (prefix << 2)
            dut.prog_tokens.value = int(W_good[i,has_connection.indices[j]])
            await ClockCycles(clock, 1)

            # write bad token weight
            dut.instruction.value = 0b01 | (prefix << 2)
            dut.prog_tokens.value = int(W_bad[i,has_connection.indices[j]])
            await ClockCycles(clock, 1)

    dut.instruction.value = 0b000
    dut.prog_tokens.value = 0
    dut.processor_id.value = 0
    dut.connection_id.value = 0
    dut.prog_tokens.value = 0
    await ClockCycles(clock, 1)


async def program(clock, dut, goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration):
    # program the network
    await program_network(dut.clock_fast, dut, W_good, W_bad, prefix=0b11)
    # program the processors
    await program_processor(dut.clock_fast, dut, goodTokensThreshold, badTokensThreshold, duration, prefix=0b10)

def parse_verilog_array(varray, s=slice(None)):
    if s == slice(None):
        s = slice(0, len(varray), 1)

    idx = range(s.start, s.stop, s.step)
    return np.array([varray[i].value.integer for i in idx])

def to_n_bit_twos_complement(num, nbit):
    if num >= 0:
        return num & ((1 << nbit) - 1)
    else:
        return ((1 << nbit) + num) & ((1 << nbit) - 1)

async def program_via_interface(clock, dut, goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration):
    NUM_PROCESSORS = int(W_good.shape[0])
    NUM_CONNECTIONS = int(dut.ttt.NUM_CONNECTIONS)
    assert W_good.shape == W_bad.shape == (NUM_PROCESSORS, NUM_PROCESSORS), "shape of W_good and W_bad must be equal to ({}, {})".format(NUM_PROCESSORS, NUM_PROCESSORS)

    # Transpose W_good and W_bad, because the hardware expects the rows to be the source and the columns to be the target
    W_good = W_good.T
    W_bad = W_bad.T

    # get the sparse connectivity
    has_connection = sps.csr_matrix(np.logical_or(W_good != 0, W_bad != 0))
    assert has_connection.nnz <= NUM_CONNECTIONS, "network has {} connections, but only {} are available".format(has_connection.nnz, NUM_CONNECTIONS)

    dut._log.info("programming network ...")

    # wait one clock cycle for safety
    await ClockCycles(clock, 1)

    # push context
    old_ui_in = dut.ui_in.value
    old_uio_in = dut.uio_in.value
    old_ena = dut.ena.value
    old_clk = dut.clk.value
    old_rst_n = dut.rst_n.value

    dut.rst_n.value = 1

    # program in all the weights

    # write all processor-indexed data
    for i in range(NUM_PROCESSORS+1):
        if i < NUM_PROCESSORS:
            # write duration
            dut.ui_in.value = (0b1001 << 4) | to_n_bit_twos_complement(i, 4)
            dut.uio_in.value = int(duration[i])
            await ClockCycles(clock, 1)

            # write good token threshold
            dut.ui_in.value = (0b1010 << 4) | to_n_bit_twos_complement(i, 4)
            dut.uio_in.value = int(goodTokensThreshold[i])
            await ClockCycles(clock, 1)

            # write bad token threshold
            dut.ui_in.value = (0b1011 << 4) | to_n_bit_twos_complement(i, 4)
            dut.uio_in.value = int(badTokensThreshold[i])
            await ClockCycles(clock, 1)

        # write indptr for processor i
        dut.ui_in.value = (0b1110 << 4) | to_n_bit_twos_complement(i, 4)
        dut.uio_in.value = int(has_connection.indptr[i])
        await ClockCycles(clock, 1)
        #print(dut.ui_in.value.binstr, dut.uio_in.value.integer, int(has_connection.indptr[i]))

    # write all connection-indexed data
    for i,(frm,to) in enumerate(zip(has_connection.indptr[:-1], has_connection.indptr[1:])):
        for j in range(frm, to):
            # set connection id
            dut.uio_in.value = j

            # write index
            dut.ui_in.value = (0b1111 << 4) | to_n_bit_twos_complement(int(has_connection.indices[j]), 4)
            await ClockCycles(clock, 1)

            # write good token weight
            dut.ui_in.value = (0b1100 << 4) | to_n_bit_twos_complement(int(W_good[i,has_connection.indices[j]]), 4)
            await ClockCycles(clock, 1)

            # write bad token weight
            dut.ui_in.value = (0b1101 << 4) | to_n_bit_twos_complement(int(W_bad[i,has_connection.indices[j]]), 4)
            await ClockCycles(clock, 1)

    # pop context
    dut.ui_in.value = old_ui_in
    dut.uio_in.value = old_uio_in
    dut.ena.value = old_ena
    dut.clk.value = old_clk
    dut.rst_n.value = old_rst_n

    # wait one clock cycle for safety
    await ClockCycles(clock, 1)


@cocotb.test()
async def test_ticktocktoken_programming(dut):
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

    # generate the weights
    while True:
        W_good = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        W_bad = np.random.poisson(0.1, (NUM_PROCESSORS, NUM_PROCESSORS)).astype(int)
        
        has_connection = sps.csr_matrix(np.logical_or(W_good.T != 0, W_bad.T != 0))

        if has_connection.nnz <= NUM_CONNECTIONS and has_connection.nnz > 10 and W_good.max() < 5 and W_bad.max() < 5:
            break

    # generate parameters
    goodTokensThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    badTokensThreshold = np.random.poisson(0.25, (NUM_PROCESSORS,)).astype(int)
    duration = np.random.poisson(5, (NUM_PROCESSORS,)).astype(int)

    # program the simulated hardware
    await program_via_interface(dut.clk, dut, goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration)
    await ClockCycles(clock, 1)

    # check if the weights were programmed in correctly
    _indptr = parse_verilog_array(dut.ttt.main.net.tgt_indptr)
    _indices = parse_verilog_array(dut.ttt.main.net.tgt_indices, slice(0, _indptr[-1], 1))
    _new_good_tokens = parse_verilog_array(dut.ttt.main.net.tgt_new_good_tokens, slice(0, _indptr[-1], 1))
    _new_bad_tokens = parse_verilog_array(dut.ttt.main.net.tgt_new_bad_tokens, slice(0, _indptr[-1], 1))
    
    # test if the processors were programmed correctly
    _good_tokens_threshold = parse_verilog_array(dut.ttt.main.proc.good_tokens_threshold)
    _bad_tokens_threshold = parse_verilog_array(dut.ttt.main.proc.bad_tokens_threshold)
    _duration = parse_verilog_array(dut.ttt.main.proc.duration)

    row_idx,col_idx = has_connection.nonzero()

    # make sure the correct values were programmed in
    assert len(_indices) == has_connection.nnz, "number of connections does not match (observed {} != {})".format(len(_indices), has_connection.nnz)
    assert np.all(_indptr == has_connection.indptr), "indptr not programmed correctly (observed {} != {})".format(_indptr, has_connection.indptr)
    assert np.all(_indices == has_connection.indices), "indices not programmed correctly (observed {} != {})".format(_indices, has_connection.indices)
    assert np.all(_new_good_tokens == [W_good.T[r,c] for r,c in zip(row_idx,col_idx)]), "good token weights not programmed correctly (observed {} != {})".format(_new_good_tokens, [W_good.T[r,c] for r,c in zip(row_idx,col_idx)])
    assert np.all(_new_bad_tokens == [W_bad.T[r,c] for r,c in zip(row_idx,col_idx)]), "bad token weights not programmed correctly (observed {} != {})".format(_new_bad_tokens, [W_bad.T[r,c] for r,c in zip(row_idx,col_idx)])

    # make sure the correct values were programmed in
    assert np.all(_good_tokens_threshold == goodTokensThreshold), "good token threshold not programmed correctly (observed {} != {})".format(_good_tokens_threshold, goodTokensThreshold)
    assert np.all(_bad_tokens_threshold == badTokensThreshold), "bad token threshold not programmed correctly (observed {} != {})".format(_bad_tokens_threshold, badTokensThreshold)
    assert np.all(_duration == duration), "duration not programmed correctly (observed {} != {}) for neuron {}".format(_duration, duration)


@cocotb.test()
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
    goodTokensThreshold = np.random.poisson(1, (NUM_PROCESSORS,)).astype(int)
    badTokensThreshold = np.random.poisson(0.25, (NUM_PROCESSORS,)).astype(int)
    duration = np.random.poisson(5, (NUM_PROCESSORS,)).astype(int)

    # make sure that processor 0, which is written first, is excitable
    goodTokensThreshold[0] = 1
    badTokensThreshold[0] = 1
    duration[0] = 2

    # program the simulated hardware
    await program_via_interface(dut.clk, dut, goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration)


    # create a golden reference model
    golden = PyTTT(goodTokensThreshold, badTokensThreshold, W_good, W_bad, duration)

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
