import cocotb
import numpy as np

def to_n_bit_twos_complement(num, nbit):
    if num >= 0:
        return num & ((1 << nbit) - 1)
    else:
        return ((1 << nbit) + num) & ((1 << nbit) - 1)

def parse_verilog_array(varray, s=slice(None)):
    if s == slice(None):
        s = slice(0, len(varray), 1)

    idx = range(s.start, s.stop, s.step)
    return np.array([varray[i].value.integer for i in idx])


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
        "{}Mismatch at ({}): {} != {}".format(
            " "*indent,
            ",".join(l),
            mat1[coords],
            mat2[coords]
        ) for (l, coords) in zip(zip(*concrete_labels), zip(*nz_indices))
    ])
