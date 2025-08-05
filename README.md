# Traffic_RL2
**Reinforcement Learning for Intelligent Traffic Signal Control**

## Overview

Traffic_RL2 implements a Deep Q-Network (DQN) based solution to optimize traffic signal timing in a simulated city intersection. The system outperforms traditional static and demand-based approaches by adapting in real time to congestion levels using reinforcement learning.

This project was built using:
- PyTorch for DQN implementation
- CityFlow for realistic traffic simulation
- Docker for environment setup and reproducibility

## Motivation

According to the Texas A&M Transportation Institute, traffic light delays can account for **12–55%** of total commute time. Most traffic lights still rely on static timers, which fail to adapt to real-world traffic patterns. Our goal is to develop an intelligent traffic control agent that:
- Reacts to live traffic conditions
- Learns optimal light switching strategies over time
- Reduces overall travel time and vehicle queue lengths

## Features

- Realistic intersection environment using CityFlow
- Modular DQN agent implemented with experience replay and target networks
- Support for multiple reward functions:
  - Pressure-based
  - Count-based
  - Hybrid (count + pressure)
- Evaluation based on average travel time, intersection throughput, and queue length

## Getting Started

### Prerequisites

- Docker
- Python 3.8+
- PyTorch
- CityFlow simulation engine

> Clone this repository and install the required dependencies:

```bash
git clone https://github.com/RestorationDev/Traffic_RL2.git
cd Traffic_RL2
pip install -r requirements.txt
```

### Running the Simulation

You can start the simulation and training using the following command:

```bash
python main.py
```

Make sure CityFlow is properly configured and available in your environment (Docker recommended for reproducibility).

### Training Details

- Algorithm: Deep Q-Network (DQN)
- Discount factor (γ): 0.99
- Batch size: 64
- Learning rate: 0.001
- Epsilon-greedy exploration: ε = 0.1

## Reward Functions

| Type | Description |
|------|-------------|
| Pressure-Based | Minimizes imbalance between incoming/outgoing vehicles |
| Count-Based | Penalizes long queues at intersections |
| Combined | Blends pressure and count with weight α = 0.5 |

## Directory Structure

```
Traffic_RL2/
├── cityflow_config/         # Traffic simulation configs
├── models/                  # Neural network models
├── main.py                  # Training entry point
├── replay_buffer.py         # Experience replay buffer
├── train.py                 # DQN training logic
├── requirements.txt
└── README.md
```

## Results

Our trained agent was able to:
- Clear all generated traffic in ~3600 seconds (vs. 4200s for baseline)
- Learn optimal signal switching strategies
- Improve both intersection throughput and vehicle wait times

## Future Work

- Expand from 1×1 to multi-intersection networks
- Integrate sensor-driven real-time data
- Explore other RL algorithms like A3C or PPO

## Authors

- Kunal Kulkarni – Model, training loop, API integration
- Tianlin Yue – Reward functions, DQN tuning
- Afnaan Qasim – State/action design, visualizations
- Max Wimmer – Documentation, research support

## License

This project is licensed under the MIT License.

---

Revolutionizing urban mobility with reinforcement learning.
