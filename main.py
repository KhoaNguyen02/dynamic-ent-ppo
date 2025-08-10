import argparse

import gymnasium

from config import *
from network import *
from ppo import *


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, default='ppo', help='ppo or de-ppo')
    parser.add_argument('--env', dest='env', type=str, default='', help='Gymnasium environment id')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='Select a seed for reproducibility')
    parser.add_argument('--device', dest='device', type=str, default='cpu', help='Select device for training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()

    # Environment configuration
    config = {
        'Pendulum-v1': PendulumConfig(args),
        'CartPole-v1': CartPoleConfig(args),
        'Acrobot-v1': AcrobotConfig(args),
        'LunarLander-v2': LunarLanderContinuousConfig(args),
        'BipedalWalker-v3': BipedalWalkerConfig(args),
    }

    # Environment setup
    envs = {
        'Pendulum-v1': gymnasium.make('Pendulum-v1'),
        'CartPole-v1': gymnasium.make('CartPole-v1'),
        'Acrobot-v1': gymnasium.make('Acrobot-v1'),
        'MountainCarContinuous-v0': gymnasium.make('MountainCarContinuous-v0', goal_velocity=0.5),
        'LunarLander-v2': gymnasium.make('LunarLander-v2', continuous=True),
        'BipedalWalker-v3': gymnasium.make('BipedalWalker-v3'),
    }
    env = envs[args.env]
    env_name = args.env

    # Set training device
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')
    print(f"Training on device: {args.device}")

    # Algorithm selection and training
    if args.type == 'ppo':
        ppo = PPO(env, ActorNetwork, CriticNetwork, config[env_name], args.device)
        ppo.learn(config[env_name].total_timesteps)
    elif args.type == 'de-ppo':
        ppo = DEPPO(env, ActorNetwork, CriticNetwork, config[env_name], args.device)
        ppo.learn(config[env_name].total_timesteps)
    else:
        raise ValueError("Invalid algorithm type. Please choose from 'ppo' or 'curl-ppo'.")
