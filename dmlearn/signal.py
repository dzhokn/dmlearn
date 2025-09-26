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