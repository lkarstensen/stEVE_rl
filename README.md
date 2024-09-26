# stEVE_rl

This is a reinforcement learning library implemented for the use with the [stEVE](https://github.com/lkarstensen/stEVE) framework. Albeit it can be used with any environment implementing the Farama gymnasium interface. 

This framework implements the Soft-Actor-Critic method using pytorch. 

The emphasis of this library lies on parallelisation of working agents, which perform episodes in the simulation and a single training agent utilizing one GPU. This is specifically helpful for computational intensive simulations. 

## Getting Started

1. Setup [stEVE](https://github.com/lkarstensen/stEVE?tab=readme-ov-file#getting-started) (including Sofa)
2. Install stEVE_rl package
   ```
   python3 -m pip install -e .
   ```
3. Test the installation
    ```
    python3 examples/function_check.py
    ```

## How to use

1. Design your Neural Network using network components (e.g. MLP, CNN, LSTM). This will define the hidden layers 
2. Bundle them in a network structure (e.g. Q-Network, Gaussian-Policy-Network). This will define the input, output and connection between the components.
3. Define Optimizer and Scheduler for the neural networks.
4. Bundle all of them in a neural network model (specific to each algorithm).
5. Define the algorithm (e.g. SAC).
6. Define a Replay Buffer. 
7. Define an Agent. 
8. Write your training loop or use one of the runners. 

Have a look at the *example* folder! More sophisticated usage examples can be found in [stEVE_training](https://github.com/lkarstensen/stEVE_training).
