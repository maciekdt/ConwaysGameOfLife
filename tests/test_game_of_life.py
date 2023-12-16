import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model.game_of_life import GameOfLife

def test_step():
    game = GameOfLife(3, device='cpu')
    # Create a simple blinker pattern
    blinker = torch.tensor([[0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0]])
    game.set_board(blinker)
    game.step()
    # After one step, the blinker should change orientation
    expected = torch.tensor([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])
    assert torch.equal(game.get_board(), expected)

    # Test another step
    game.step()
    assert torch.equal(game.get_board(), blinker)