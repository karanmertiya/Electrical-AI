# Electrical-AI: Neuromorphic Spiking Neural Network (SNN)

This repository implements a **Spiking Neural Network (SNN)** using Leaky Integrate-and-Fire (LIF) neuron dynamics, designed to demonstrate extreme thermodynamic efficiency for Edge AI hardware (e.g., BrainChip Akida, Intel Loihi).

## Architecture & Physics
Unlike standard Artificial Neural Networks (ANNs) which perform dense, floating-point matrix multiplications at every layer, SNNs operate on sparse, binary spikes distributed over time. 

This repository physically counts the binary spikes emitted during the temporal sequence to calculate the **Spike Sparsity**. High spike sparsity directly correlates to massive reductions in power consumption on physical neuromorphic hardware, as transistors only draw dynamic power when a spike event occurs.

## Execution
The network is built using `snntorch` and trains on the MNIST dataset, proving the network converges mathematically while maintaining extreme sparsity.

## Requirements
```
torch
torchvision
snntorch
```

## Running the Pipeline
```bash
python snn_pipeline.py
```
