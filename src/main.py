import os
import torch
from view.game_of_life_visualizer import GameOfLifeVisualizer
from network.basic_cnn import LitSimpleCNN

#$env:PYTHONPATH="C:\Users\maciek\Documents\ConwaysGameOfLife"

model = LitSimpleCNN.load_from_checkpoint("saved_models/model_checkpoint.ckpt")

data = torch.tensor([[0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]])

data = data.unsqueeze(0).unsqueeze(0)

model.eval()
with torch.no_grad():
    prediction = model(data)
    
processed_prediction = torch.round(prediction).squeeze(0)

visualizer = GameOfLifeVisualizer()
visualizer.display_board(processed_prediction)