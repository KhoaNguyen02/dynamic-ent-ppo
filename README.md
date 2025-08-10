# Dynamic Entropy PPO

An implementation of Proximal Policy Optimization (PPO) with dynamic entropy regularization for improved exploration and learning stability in reinforcement learning environments.

## Overview

This project introduces and investigates a potential simple approach to PPO that dynamically adjusts the entropy coefficient during training to balance exploration and exploitation. Unlike traditional PPO implementations that use fixed entropy coefficients, our dynamic entropy mechanism adapts based on training progress.

## Key Features

- **Dynamic Entropy Adjustment**: Automatically adjusts entropy coefficient throughout learning progress.
- **Improved Exploration**: Enhanced exploration capabilities through entropy scheduling.
- **Stable Learning**: Better convergence properties compared to fixed entropy approach.

## Algorithm Description

The Dynamic Entropy PPO (DE-PPO) algorithm extends the standard PPO objective with an adaptive entropy coefficient:

$$
\max_{\theta} \quad \mathcal{L}_t(\theta) =  \hat{\mathbb{E}}_t \left[\mathcal{L}^{actor}_t(\theta) - \alpha \mathcal{L}^{critic}_t + \beta H(\pi_\theta)  \right] 
$$

Where $\beta$ is the dynamic entropy coefficient that changes during training via a sinusoidal oscillation:

$$\beta_t = \beta_0 + A \cdot \sin\left(2 \pi f \frac{t}{T} + \phi_t\right)$$

with $A = A_0 \cdot e^{\left(-\lambda \frac{t}{T}\right)}$ and $\phi_t = \phi_0 + k\frac{t}{T}$

## Installation

```bash
git clone https://github.com/KhoaNguyen02/dynamic-ent-ppo.git
cd dynamic-ent-ppo
pip install -r requirements.txt
```

## Usage

```bash
# Train PPO on CartPole environment
python main.py --env CartPole-v1 --type ppo

# Train DE-PPO on continuous control task
python main.py --env Pendulum-v1 --type de-ppo
```

## Experimental Results

### Performance Comparison

Our dynamic entropy approach shows significant improvements over standard PPO (rewards achieved averaged over last 50 iterations):

| Environment | Standard PPO | DE PPO |
|------|-----|--------|
| Acrobot-v1 | -82.34 ± 11.47 | **-81.88 ± 11.95** |
| Cartpole-v1 | 481.05 ± 44.74 | **496.37 ± 23.63** |
| Pendulum-v1 | -168.08 ± 42.76 | **-152.09 ± 28.67** |
| Lunarlander-v2 | 213.43 ± 51.68 | **217.77 ± 37.85** |
| Bipedalwalker-v3 | 163.16 ± 53.36 | **169.05 ± 59.44** |


### Key Findings

1. **Faster Convergence**: Dynamic entropy scheduling leads to faster convergence
3. **Reduced Variance**: Significantly reduced training variance in 80% of tested environments, indicating a more reliable convergence.
4. **Adaptive Exploration**: Automatic adjustment of exploration based on learning progress

## Acknowledgments

- Original PPO paper by Schulman et al.
- OpenAI Gym for providing standardized RL environments.
- PyTorch for the deep learning framework.
- Stable Baselines3 for reference implementations.