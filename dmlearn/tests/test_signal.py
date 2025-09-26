import numpy as np
from dmlearn.signal import pad_tensor

def test_pad_tensor_2D_offset_1():
    """Test the pad_tensor function with various inputs."""
    # Test case 1: 2D tensor with padding_offset=1 and default padding_fill_value
    tensor_2d = np.array([[1, 2], [3, 4]])
    expected_2d = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 0], 
        [0, 3, 4, 0],
        [0, 0, 0, 0]
    ])
    padded_2d = pad_tensor(tensor_2d, padding=1)
    np.testing.assert_array_equal(padded_2d, expected_2d)

def test_pad_tensor_2D_offset_2():
    tensor_2d = np.array([[1, 2], [3, 4]])
    # Test case 2: 2D tensor with padding_offset=2 and custom padding_fill_value
    expected_2d_custom = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    padded_2d_custom = pad_tensor(tensor_2d, padding=2)
    np.testing.assert_array_equal(padded_2d_custom, expected_2d_custom)

def test_pad_tensor_2D_padding_fill_value_9():
    tensor_2d = np.array([[1, 2], [3, 4]])
    expected_2d_custom = np.array([
        [9, 9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9, 9],
        [9, 9, 1, 2, 9, 9],
        [9, 9, 3, 4, 9, 9],
        [9, 9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9, 9]
    ])
    padded_2d_custom = pad_tensor(tensor_2d, padding=2, fill_value=9)
    np.testing.assert_array_equal(padded_2d_custom, expected_2d_custom)

def test_pad_tensor_3D_offset_1():
    tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected_3d = np.array([
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 5, 6, 0], [0, 7, 8, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ])
    padded_3d = pad_tensor(tensor_3d, padding=1)
    np.testing.assert_array_equal(padded_3d, expected_3d)

def test_pad_tensor_3D_offset_2():
    tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected_3d = np.array([
        [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 0, 0], [0, 0, 3, 4, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 5, 6, 0, 0], [0, 0, 7, 8, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    ])
    padded_3d = pad_tensor(tensor_3d, padding=2)
    np.testing.assert_array_equal(padded_3d, expected_3d)

def test_pad_tensor_3D_padding_fill_value_9():
    tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected_3d = np.array([
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 1, 2, 9], [9, 3, 4, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 5, 6, 9], [9, 7, 8, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]
    ])
    padded_3d = pad_tensor(tensor_3d, padding=1, fill_value=9)
    np.testing.assert_array_equal(padded_3d, expected_3d)

