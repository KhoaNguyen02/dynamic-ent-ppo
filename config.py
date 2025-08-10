import os
import numpy as np


class BaseConfig:
    def __init__(self, args):
        self.n_steps = 2048  # Number of steps per environment per update
        self.max_timesteps = 200 # Maximum number of timesteps per episode
        self.n_epochs = 10  # Number of epochs per update
        self.lr = 3e-4  # Learning rate
        self.total_timesteps = 1_000_000 # Total number of simulation timesteps

        self.batch_size = 64  # Minibatch size
        self.gamma = 0.99  # Discount factor
        self.lambd = 0.98  # GAE lambda
        self.clip = 0.2  # PPO clipping parameter
        self.vt_coef = 0.5  # Value loss coefficient
        self.ent_coef = 0.0  # Entropy coefficient
        self.target_kl = 0.01  # Target KL divergence
        self.max_grad_norm = 0.5  # Gradient clipping
        self.save_freq = 10
        self.render = False
        self.seed = args.seed

class AcrobotConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.env_name = 'acrobot-v1'
        self.n_steps = 1024
        self.max_timesteps = 500
        self.n_epochs = 10
        self.lr = 3e-4
        self.total_timesteps = 500_000

        # DE-PPO specific parameters
        self.base_phase = 0.0
        self.phase_slope = 1e-4
        self.ent_coef_amp = 1e-4
        self.lambda_decay = 0.2
        self.ent_coef_freq = 2.0
        self.ent_coef_base = 0.0

        # Directory setup
        os.makedirs(f'saved/logs/{self.env_name}/ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/logs/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/ppo', exist_ok=True)


class CartPoleConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.env_name = 'cartpole-v1'
        self.n_steps = 1024
        self.max_timesteps = 500
        self.n_epochs = 20
        self.lr = 2e-4
        self.total_timesteps = 500_000

        # DE-PPO specific parameters
        self.base_phase = 0.0
        self.phase_slope = 0.3
        self.ent_coef_amp = 0.05
        self.lambda_decay = 0.01
        self.ent_coef_freq = 1.0
        self.ent_coef_base = 0.01

        # Directory setup
        os.makedirs(f'saved/logs/{self.env_name}/ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/logs/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/ppo', exist_ok=True)


class PendulumConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.env_name = 'pendulum-v1'
        self.n_steps = 2048
        self.max_timesteps = 200
        self.n_epochs = 10
        self.lr = 3e-4
        self.total_timesteps = 1_000_000

        # DE-PPO specific parameters
        self.base_phase = np.pi / 2
        self.phase_slope = 1.0
        self.ent_coef_amp = 0.1
        self.lambda_decay = 0.01
        self.ent_coef_freq = 1.0
        self.ent_coef_base = 0.01

        # Directory setup
        os.makedirs(f'saved/logs/{self.env_name}/ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/logs/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/ppo', exist_ok=True)


class LunarLanderContinuousConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.env_name = 'lunarlander-v2'
        self.n_steps = 2048
        self.max_timesteps = 1000
        self.n_epochs = 10
        self.lr = 3e-4
        self.total_timesteps = 1_500_000
        self.ent_coef = 0.01

        # DE-PPO specific parameters
        self.base_phase = np.pi
        self.phase_slope = 1.5
        self.ent_coef_amp = 0.3
        self.lambda_decay = 0.1
        self.ent_coef_freq = 0.5
        self.ent_coef_base = 0.05

        # Directory setup
        os.makedirs(f'saved/logs/{self.env_name}/ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/logs/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/ppo', exist_ok=True)


class BipedalWalkerConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.env_name = 'bipedalwalker-v3'
        self.n_steps = 4096
        self.max_timesteps = 1600
        self.n_epochs = 10
        self.lr = 3e-4
        self.total_timesteps = 1_500_000

        # DE-PPO specific parameters
        self.base_phase = 0.0
        self.phase_slope = 0.1
        self.ent_coef_amp = 0.005
        self.lambda_decay = 0.005
        self.ent_coef_freq = 0.05
        self.ent_coef_base = 0.001

        # Directory setup
        os.makedirs(f'saved/logs/{self.env_name}/ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/logs/{self.env_name}/de-ppo', exist_ok=True)
        os.makedirs(f'saved/models/{self.env_name}/ppo', exist_ok=True)
