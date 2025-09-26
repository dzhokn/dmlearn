import numpy as np

def pad_tensor(
        tensor: np.ndarray, 
        padding_offset: int, 
        padding_fill_value: float = 0.0) -> np.ndarray:
    """
    Pad a 2D or 3D tensor with a given value. 
    e.g. [[1,2], [3,4]] -> [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]

    Args:
        tensor: The tensor to pad (e.g. [[1,2], [3,4]])
        padding_offset: The number of elements to pad on each side (e.g. 1)
        padding_fill_value: The value to pad the tensor with (default value is 0)

    Returns:
        The padded tensor (e.g. [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    """
    # CASE 1: 2D tensor
    if tensor.ndim == 2:
        return np.pad(tensor, ((padding_offset, padding_offset), (padding_offset, padding_offset)), mode='constant', constant_values=padding_fill_value)
    # CASE 2: 3D tensor
    else:
        return np.pad(tensor, ((padding_offset, padding_offset), (padding_offset, padding_offset), (padding_offset, padding_offset)), mode='constant', constant_values=padding_fill_value)