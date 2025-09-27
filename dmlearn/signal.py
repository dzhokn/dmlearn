import numpy as np

def pad_tensor(
        tensor: np.ndarray,
        padding: int,
        fill_value: float = 0.0) -> np.ndarray:
    """
    Pad a 2D or 3D tensor with a given value.
    e.g. [[1,2], [3,4]] -> [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]

    Args:
        tensor: The tensor to pad (e.g. [[1,2], [3,4]])
        padding: The number of elements to pad on each side (e.g. 1)
        fill_value: The value to pad the tensor with (default value is 0)

    Returns:
        The padded tensor (e.g. [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    """
    # CASE 1: 2D tensor
    if tensor.ndim == 2:
        return np.pad(tensor, ((padding, padding), (padding, padding)), mode='constant', constant_values=fill_value)
    # CASE 2: 3D tensor
    else:
        return np.pad(tensor, ((padding, padding), (padding, padding), (padding, padding)), mode='constant', constant_values=fill_value)

def chunk_tensor(
        tensor: np.ndarray,
        chunk_size: int,
        step_size: int) -> np.ndarray:
    """
    Convert a 2D or 3D tensor to strided (overlapping) chunks. A chunk is expected to be a square matrix (e.g. 2x2, 3x3, etc.).
    e.g. [[1, 2, 3], [4, 5, 6], [7, 8, 9]] -> [[[1, 2], [4, 5]], [[2, 3], [5, 6]], [[4, 5], [7, 8]], [[5, 6], [8, 9]]]

    Args:
        tensor:                 The tensor to convert (e.g. [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        chunk_size:             The size of the output chunks (e.g. chunk_size=2 means that the chunks will be 2x2 matrices)
        step_size (stride):     The number of elements after which next chunk starts (e.g. step_size=1 means that the next chunk starts after 1 element)

    Returns:
        The chunked tensor (e.g. [[[0, 1], [4, 5]], [[1, 2], [5, 6]], [[2, 3], [6, 7]], [[1, 1], [0, 0]]])
        The output shape for 2D tensor is (num_of_chunks, chunk_rows, chunk_cols)
        The output shape for 3D tensor is (tensor.shape[0], num_of_chunks, chunk_rows, chunk_cols)
    """
    # CASE 1: 2D tensor - (num_of_chunks, chunk_size, chunk_size)
    if tensor.ndim == 2:
        return _chunk_tensor_2d(tensor, chunk_size, step_size)
    # CASE 2: 3D tensor - (tensor depth, num_of_chunks, chunk_size, chunk_size)
    else:
        return _chunk_tensor_3d(tensor, chunk_size, step_size)

def _chunk_tensor_2d(tensor: np.ndarray, chunk_size: int, step_size: int) -> np.ndarray:
    # EXPLANATION HOW NUMPY STRIDES (BYTE MARKERS) ARE CALCULATED:
    # An instance of class ndarray consists of a contiguous one-dimensional segment of computer memory (owned by the array, or by some other object),
    # combined with an indexing scheme that maps N integers into the location of an item in the block. The strides of an array tell us how many bytes
    # we have to skip in memory to move to the next position along a certain axis. For example, in the tensor [[1, 2, 3], [4, 5, 6], [7, 8, 9]] we
    # have to skip 8 bytes (1 value) to move to the next column, but 24 bytes (3 values) to get to the same position in the next row.
    byte_offset_row = tensor.strides[0] # In order to move to the value in the next row, we have to skip this number of bytes (e.g. 24 bytes)
    byte_offset_col = tensor.strides[1] # In order to move to the value in the next column, we have to skip this number of bytes (e.g. 8 bytes)

    # Calculate the number of all chunks (e.g. 2, 2, 3, 3 - 2x2 times move a 3x3 window across the tensor, the 3x3 is the chunk size):
    num_of_chunks_in_row, num_of_chunks_in_col = __calculate_num_of_chunks(tensor, chunk_size, step_size)
    chunked_tensor_shape = (num_of_chunks_in_row, num_of_chunks_in_col, chunk_size, chunk_size)

    # Calculate the strides of the chunks (e.g. (24, 8, 8, 8) for tensor [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # - step_size*byte_offset_row   : the number of bytes to move down to the next chunk in that row
    # - step_size*byte_offset_col   : the number of bytes to move right to the next chunk in that column
    # - byte_offset_row             : the number of bytes to move to the next element in that row
    # - byte_offset_col             : the number of bytes to move to the next element in that column
    strides = (step_size*byte_offset_row, step_size*byte_offset_col, byte_offset_row, byte_offset_col)

    # Slice the tensor to get the chunks
    # - tensor              : the tensor to convert
    # - chunked_tensor_shape: the shape of the tensor, containing all the output chunks
    # - strides             : the strides of the chunks
    chunks = np.lib.stride_tricks.as_strided(x = tensor, shape = chunked_tensor_shape, strides = strides)

    # Reshape the output to (num_of_chunks, chunk_size, chunk_size). Basically, flatten the first two dimensions of the output tensor.
    return chunks.reshape(-1, chunk_size, chunk_size)

def _chunk_tensor_3d(tensor: np.ndarray, chunk_size: int, step_size: int) -> np.ndarray:
    # Strides are the number of bytes to step in each dimension when traversing an array.
    byte_offset_depth, byte_offset_row, byte_offset_col = tensor.strides
    # Calculate the number of all chunks (e.g. 3, 2, 2, 2, 2 - 3x2x2 times move a 2x2 window across the tensor, the 2x2 is the chunk size):
    tensor_depth = tensor.shape[0]
    num_of_chunks_in_row, num_of_chunks_in_col = __calculate_num_of_chunks(tensor, chunk_size, step_size)
    chunked_tensor_shape = (tensor_depth, num_of_chunks_in_row, num_of_chunks_in_col, chunk_size, chunk_size)
    # Calculate the strides of the chunks (e.g. (72, 24, 8, 24, 8) for tensor [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    strides = (byte_offset_depth, step_size*byte_offset_row, step_size*byte_offset_col, byte_offset_row, byte_offset_col)
    # Slice the tensor to get the chunks
    chunks = np.lib.stride_tricks.as_strided(x = tensor, shape = chunked_tensor_shape, strides = strides)
    # Reshape the output to (input_tensor_depth,num_of_chunks, chunk_size, chunk_size). Basically, flatten the second and third dimensions of the output tensor.
    return chunks.reshape(tensor_depth, -1, chunk_size, chunk_size)

def __calculate_num_of_chunks(tensor: np.ndarray, chunk_size: int, step_size: int) -> tuple[int, int]:
    # Calculate the number of chunks within a single tensor matrix:
    # - 1+size_diff // step_size: the number of chunks in a single matrix row
    # - 1+size_diff // step_size: the number of chunks in a single matrix column
    num_of_chunks_in_row = 1 + (tensor.shape[0] - chunk_size) // step_size
    num_of_chunks_in_col = 1 + (tensor.shape[1] - chunk_size) // step_size
    # If the number of chunks in a row or column is 0, set it to 1 (we always have at least one chunk)
    num_of_chunks_in_row = max(1, num_of_chunks_in_row)
    num_of_chunks_in_col = max(1, num_of_chunks_in_col)
    # Return the number of chunks in a row and the number of chunks in a column
    return num_of_chunks_in_row, num_of_chunks_in_col