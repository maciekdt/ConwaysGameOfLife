import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model.board_initializer import BoardInitializer

def test_numbers_to_input_row_basic():
    # Test with basic numbers
    result = BoardInitializer.numbers_to_input_row(num1=2, num2=3, row_length=20, divider_length=3)
    expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float16)
    assert torch.equal(result, expected)

def test_numbers_to_input_row_to_small_row_length():
    with pytest.raises(ValueError):
        BoardInitializer.numbers_to_input_row(num1=2, num2=3, row_length=18, divider_length=3)


def test_output_row_to_tensor_basic():
    imput_row = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float16), 