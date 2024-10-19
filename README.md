# Predicting Conway's Game of Life Using a Convolutional Neural Network
This project was born out of curiosity: can a neural network handle the chaotic and complex nature of **Conway's Game of Life**?
As a cellular automaton that is Turing complete, Conway's Game of Life has fascinated researchers with its potential to simulate computation.
Inspired by concepts such as *Conway's Game within Conway's Game*—where structures are crafted by humans to perform computations—I wondered if a neural network could independently discover simple 
computational machines in this environment. 

However, this project starts simpler: focusing first on how a network can predict the evolution of objects in the Game of Life over time.
Given the unpredictable nature of this environment, it's no easy task.

## Methodology
I employed a **convolutional neural network** built using **PyTorch**, trained on randomly generated samples from a 7x7 grid.
These grids were evolved forward by 20 steps, with the network tasked with predicting the final step in the evolution.

## Data Generation Process
The process of generating dataset for training was computationally intensive, especially for larger grids or more steps.
To optimize it, I implemented a matrix-based algorithm for Conway's Game of Life. 
Generating these patterns was computationally expensive, especially as 80% of random patterns on a 7x7 board evolve into empty boards within 20 steps. 
To avoid unbalancing the dataset, I discarded around 95% of the patterns that evolved to empty boards.

The initial number of live cells on the board was sampled from a normal distribution.
The cells were clustered close together to generate more interesting, less dispersed structures.

Fragment of the dataset contains both the initial state and the evolved state after 20 steps:
![image](https://github.com/user-attachments/assets/81356770-8257-441d-aa72-11fc7f86d3da)

Histograms representing the number of live cells in both the initial and evolved states (after filtering out empty states): 
![image](https://github.com/user-attachments/assets/8909a418-0863-4b6e-bc3d-22e359c395f0)

The dataset contains approximately **1,000,000 examples**.

## Training Process 
The neural network was composed of 7 convolutional layers with increasing filter sizes, as follows: 32-32-64-128-128-128-128, followed by a fully connected layer.
The network was trained using cross-entropy loss, with training continuing until the change in loss was smaller than 0.0001 for three consecutive epochs.
Training took place over several dozen epochs on my local GPU.

## Results
While predicting the exact evolution of any arbitrary object in Conway’s Game of Life is a challenging problem, the results so far look **promising**.
The network is able to capture some of the simpler patterns and make reasonable predictions for future states.

Here are examples of some simple patterns, their evolution and CNN prediction:
#### Example 1 - the simplest stable patterns:
![image](https://github.com/user-attachments/assets/47093d44-8675-4180-b55e-b1de09d03686)
#### Example 2 - more complex stable patterns:
![image](https://github.com/user-attachments/assets/b8a59f75-f1c5-49e4-adfa-237a371d00f3)
#### Example 3 - blinkers and other changable patterns:
![image](https://github.com/user-attachments/assets/bd384434-90d5-4682-abd0-73ba0bb0f776)


## Next Steps
The next phase of this research will involve applying **reinforcement learning** techniques.
The goal is to train the network to achieve specific objectives within the Game of Life, rather than simply predicting arbitrary pattern evolutions.
This will involve more complex tasks like guiding the system to build certain patterns or machines far more challenging but exciting direction.

This is just the beginning of my exploration into how neural networks interact with environments as rich and complex as the Game of Life. To push the boundaries further, it will also be important to **increase the size of the network** and **boost computational power** using cloud services.