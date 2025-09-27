import numpy as np
from dmlearn.signal import chunk_tensor, pad_tensor

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
    np.testing.assert_array_equal(padded_2d.shape, (4, 4))
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
    np.testing.assert_array_equal(padded_2d_custom.shape, (6, 6))
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
    np.testing.assert_array_equal(padded_2d_custom.shape, (6, 6))
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
    np.testing.assert_array_equal(padded_3d.shape, (4, 4, 4))
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
    np.testing.assert_array_equal(padded_3d.shape, (6, 6, 6))
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
    np.testing.assert_array_equal(padded_3d.shape, (4, 4, 4))
    np.testing.assert_array_equal(padded_3d, expected_3d)

def test_chunk_tensor_3x3_chunk_size_1_step_size_1():
    tensor_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_2d = np.array([[[1]], [[2]], [[3]],[[4]], [[5]], [[6]],[[7]], [[8]], [[9]]])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=1, step_size=1)
    np.testing.assert_array_equal(chunked_2d.shape, (9, 1, 1))
    np.testing.assert_array_equal(chunked_2d, expected_2d)

def test_chunk_tensor_3x3_chunk_size_2_step_size_1():
    tensor_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_2d = np.array([
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
        [[4, 5], [7, 8]],
        [[5, 6], [8, 9]]
    ])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=2, step_size=1)
    np.testing.assert_array_equal(chunked_2d.shape, (4, 2, 2))
    np.testing.assert_array_equal(chunked_2d, expected_2d)

def test_chunk_tensor_3x3_chunk_size_2_step_size_2():
    tensor_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_2d = np.array([[[1, 2], [4, 5]]])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=2, step_size=2)
    np.testing.assert_array_equal(chunked_2d.shape, (1, 2, 2))
    np.testing.assert_array_equal(chunked_2d, expected_2d)

def test_chunk_tensor_3x3_chunk_size_3_step_size_1():
    tensor_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_2d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=3, step_size=1)
    np.testing.assert_array_equal(chunked_2d.shape, (1, 3, 3))
    np.testing.assert_array_equal(chunked_2d, expected_2d)

def test_chunk_tensor_3x3_chunk_size_3_step_size_2():
    tensor_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_2d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=3, step_size=2)
    np.testing.assert_array_equal(chunked_2d.shape, (1, 3, 3))
    np.testing.assert_array_equal(chunked_2d, expected_2d)

def test_chunk_tensor_4x4_chunk_size_2_step_size_1():
    tensor_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    expected_2d = np.array([
        [[1, 2], [5, 6]],
        [[2, 3], [6, 7]],
        [[3, 4], [7, 8]],
        [[5, 6], [9, 10]],
        [[6, 7], [10, 11]],
        [[7, 8], [11, 12]],
        [[9, 10], [13, 14]],
        [[10, 11], [14, 15]],
        [[11, 12], [15, 16]]
    ])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=2, step_size=1)
    np.testing.assert_array_equal(chunked_2d.shape, (9, 2, 2))
    np.testing.assert_array_equal(chunked_2d, expected_2d)

def test_chunk_tensor_4x4_chunk_size_3_step_size_1():
    tensor_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    chunked_2d = chunk_tensor(tensor_2d, chunk_size=3, step_size=1)
    np.testing.assert_array_equal(chunked_2d.shape, (4, 3, 3))

def test_chunk_tensor_3x3x3_chunk_size_2_step_size_1():
    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    expected_3d = np.array([
        [
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
        [[4, 5], [7, 8]],
        [[5, 6], [8, 9]]
        ],
        [
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
        [[4, 5], [7, 8]],
        [[5, 6], [8, 9]]
        ],
        [
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
        [[4, 5], [7, 8]],
        [[5, 6], [8, 9]]
        ]
    ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=2, step_size=1)
    np.testing.assert_array_equal(chunked_3d.shape, (3, 4, 2, 2))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_3x3x3_chunk_size_2_step_size_2():
    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    expected_3d = np.array([[[[1, 2], [4, 5]]],[[[1, 2], [4, 5]]],[[[1, 2], [4, 5]]]])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=2, step_size=2)
    np.testing.assert_array_equal(chunked_3d.shape, (3, 1, 2, 2))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_3x3x3_chunk_size_3_step_size_1():
    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    expected_3d = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=3, step_size=1)
    np.testing.assert_array_equal(chunked_3d.shape, (3, 1, 3, 3))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_3x3x3_chunk_size_3_step_size_2():
    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    expected_3d = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=3, step_size=2)
    np.testing.assert_array_equal(chunked_3d.shape, (3, 1, 3, 3))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_2_step_size_1():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2], [5, 6]],
                [[2, 3], [6, 7]],
                [[3, 4], [7, 8]],
                [[5, 6], [9, 10]],
                [[6, 7], [10, 11]],
                [[7, 8], [11, 12]],
                [[9, 10], [13, 14]],
                [[10, 11], [14, 15]],
                [[11, 12], [15, 16]]
            ],
            [
                [[1, 2], [5, 6]],
                [[2, 3], [6, 7]],
                [[3, 4], [7, 8]],
                [[5, 6], [9, 10]],
                [[6, 7], [10, 11]],
                [[7, 8], [11, 12]],
                [[9, 10], [13, 14]],
                [[10, 11], [14, 15]],
                [[11, 12], [15, 16]]
            ],
            [
                [[1, 2], [5, 6]],
                [[2, 3], [6, 7]],
                [[3, 4], [7, 8]],
                [[5, 6], [9, 10]],
                [[6, 7], [10, 11]],
                [[7, 8], [11, 12]],
                [[9, 10], [13, 14]],
                [[10, 11], [14, 15]],
                [[11, 12], [15, 16]]
            ],
            [
                [[1, 2], [5, 6]],
                [[2, 3], [6, 7]],
                [[3, 4], [7, 8]],
                [[5, 6], [9, 10]],
                [[6, 7], [10, 11]],
                [[7, 8], [11, 12]],
                [[9, 10], [13, 14]],
                [[10, 11], [14, 15]],
                [[11, 12], [15, 16]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=2, step_size=1)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 9, 2, 2))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_2_step_size_2():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2], [5, 6]],
                [[3, 4], [7, 8]],
                [[9, 10], [13, 14]],
                [[11, 12], [15, 16]]
            ],
            [
                [[1, 2], [5, 6]],
                [[3, 4], [7, 8]],
                [[9, 10], [13, 14]],
                [[11, 12], [15, 16]]
            ],
            [
                [[1, 2], [5, 6]],
                [[3, 4], [7, 8]],
                [[9, 10], [13, 14]],
                [[11, 12], [15, 16]]
            ],
            [
                [[1, 2], [5, 6]],
                [[3, 4], [7, 8]],
                [[9, 10], [13, 14]],
                [[11, 12], [15, 16]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=2, step_size=2)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 4, 2, 2))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_2_step_size_3():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2], [5, 6]]
            ],
            [
                [[1, 2], [5, 6]]
            ],
            [
                [[1, 2], [5, 6]]
            ],
            [
                [[1, 2], [5, 6]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=2, step_size=3)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 1, 2, 2))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_3_step_size_1():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                [[2, 3, 4], [6, 7, 8], [10, 11, 12]],
                [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                [[6, 7, 8], [10, 11, 12], [14, 15, 16]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                [[2, 3, 4], [6, 7, 8], [10, 11, 12]],
                [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                [[6, 7, 8], [10, 11, 12], [14, 15, 16]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                [[2, 3, 4], [6, 7, 8], [10, 11, 12]],
                [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                [[6, 7, 8], [10, 11, 12], [14, 15, 16]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                [[2, 3, 4], [6, 7, 8], [10, 11, 12]],
                [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                [[6, 7, 8], [10, 11, 12], [14, 15, 16]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=3, step_size=1)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 4, 3, 3))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_3_step_size_2():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=3, step_size=2)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 1, 3, 3))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_3_step_size_3():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ],
            [
                [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=3, step_size=3)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 1, 3, 3))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_4_step_size_1():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ],
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ],
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ],
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=4, step_size=1)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 1, 4, 4))
    np.testing.assert_array_equal(chunked_3d, expected_3d)

def test_chunk_tensor_4x4x4_chunk_size_4_step_size_4():
    tensor_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    expected_3d = np.array([
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ],
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ],
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ],
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ]
        ])
    chunked_3d = chunk_tensor(tensor_3d, chunk_size=4, step_size=4)
    np.testing.assert_array_equal(chunked_3d.shape, (4, 1, 4, 4))
    np.testing.assert_array_equal(chunked_3d, expected_3d)