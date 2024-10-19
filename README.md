# Predicting Conway's Game of Life Using a Convolutional Neural Network

This project was born out of curiosity: can a neural network handle the chaotic and complex nature of **Conway's Game of Life**?
As a cellular automaton that is Turing complete, Conway's Game of Life has fascinated researchers with its potential to simulate computation.
Inspired by concepts such as *Conway's Game within Conway's Game*—where structures are crafted by humans to perform computations—I wondered if a neural network could independently discover simple 
computational machines in this environment. 

However, this project starts simpler: focusing first on how a network can predict the evolution of objects in the Game of Life over time.
Given the unpredictable nature of this environment, it's no easy task.

## Methodology
I employed a **convolutional neural network** built using **PyTorch**, trained on randomly generated samples from a 7x7 grid.
These grids were evolved forward by 20 steps, with the network tasked with predicting the next steps in the evolution.
The dataset was generated procedurally, and both training and dataset generation were optimized for GPU performance.

Here's an example of the generated data:
![image](https://github.com/user-attachments/assets/81356770-8257-441d-aa72-11fc7f86d3da)

## Results
While predicting the exact evolution of any arbitrary object in Conway’s Game of Life is a challenging problem, the results so far look **promising**.
The network is able to capture some of the simpler patterns and make reasonable predictions for future states.

Here are examples of some simple patterns, their evolution and CNN prediction:

![image](https://github.com/user-attachments/assets/47093d44-8675-4180-b55e-b1de09d03686)
![image](https://github.com/user-attachments/assets/b8a59f75-f1c5-49e4-adfa-237a371d00f3)
![image](https://github.com/user-attachments/assets/bd384434-90d5-4682-abd0-73ba0bb0f776)


## Next Steps
The next phase of this research will involve applying **reinforcement learning** techniques.
The goal is to train the network to achieve specific objectives within the Game of Life, rather than simply predicting arbitrary object evolutions.
This will involve more complex tasks like guiding the system to build certain patterns or machines—a far more challenging but exciting direction.

This is just the beginning of my exploration into how neural networks interact with environments as rich and complex as the Game of Life.
