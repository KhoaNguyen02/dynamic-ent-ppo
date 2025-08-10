import csv

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class PPO:
    def __init__(self, env, actor_class, critic_class, config, device):
        # Environment
        self.env = env
        # Training device
        self.device = device
        # Hyperparameters configuration
        self.config = config
        # Reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            self.env.reset(seed=self.config.seed)
            print(f"Training with seed: {self.config.seed}")

        # Determine if the action space is discrete or continuous
        self.discrete = isinstance(env.action_space, Discrete)
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]

        # Initialize actor and critic networks
        self.actor = actor_class(self.state_dim, self.action_dim, discrete=self.discrete).to(self.device)
        self.critic = critic_class(self.state_dim).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.config.lr)

    def learn(self, total_timesteps, log_filename='ppo_logs'):
        """Train the PPO agent on the given environment.

        Parameters
        ----------
        total_timesteps : int
            Total number of simulation steps for training.
        log_filename : str, optional
            Name of the log file to save training metrics, by default 'ppo_logs'
        """
        assert log_filename in ['ppo_logs', 'de_ppo_logs'], "log_filename must be 'ppo_logs' or 'de_ppo_logs'"

        # Actor and Critic network architectures
        print("-"*100)
        print(self.actor)
        print("-"*100)
        print(self.critic)
        print("-"*100)

        # Determine model type for logging
        model_name = log_filename.split('_')[0]
        if model_name == 'ppo':
            model_type = 'ppo'
        else:
            model_type = 'de-ppo'

        # CSV logging file
        csv_filename = f'saved/logs/{self.config.env_name}/{model_type}/{log_filename}_{self.config.seed}.csv'
        curr_timestep, curr_iter = 0, 0
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['iteration', 'timesteps', 'return', 'actor_loss', 'critic_loss', 'total_loss'])
            # PPO training loop
            with tqdm(total=total_timesteps, desc='Training Progress') as pbar:
                while curr_timestep < total_timesteps:
                    # Collect trajectories
                    states, actions, log_probs, rewards, values = self._collect_rollout()
                    # Compute advantages and returns
                    advantages = self._calculate_gae(rewards, values)
                    rewards_to_go = advantages + self.critic(states).detach()
                    # Update current timestep and iteration
                    curr_timestep += states.size(0)
                    curr_iter += 1
                    # Normalize advantages to stabilize training
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
                    # Prepare mini-batches
                    dataset = TensorDataset(states, actions, log_probs, advantages, rewards_to_go)
                    dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
                    # Update actor-critic networks
                    actor_losses, critic_losses, total_losses = self._update_networks(dataloader, total_timesteps, curr_timestep)
                    # Compute average return and actor loss
                    avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
                    avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
                    avg_total_loss = np.mean(total_losses) if total_losses else 0.0
                    avg_episode_return = np.mean([sum(ep) for ep in rewards]) if rewards else 0.0
                    # Log metrics to CSV
                    self._log_metrics(csv_file, csv_writer, pbar, curr_iter, curr_timestep, states.size(0), 
                                    avg_episode_return, avg_actor_loss, avg_critic_loss, avg_total_loss)
                    # Save models periodically
                    if curr_iter % self.config.save_freq == 0:
                        self._save_models()

    def _collect_rollout(self):
        """Run simulation for a batch of episodes and collect rollout data.

        Returns
        -------
        states : torch.Tensor
            Tensor of shape (n_steps, state_dim) containing states.
        actions : torch.Tensor
            Tensor of shape (n_steps,) containing actions.
        log_probs : torch.Tensor
            Tensor of shape (n_steps,) containing log probabilities of actions.
        rewards : list
            List of length n_episodes containing rewards per episode.
        values : list
            List of length n_episodes containing value estimates per episode.
        """
        states, actions, log_probs, rewards, values = [], [], [], [], []
        timesteps_collected = 0

        # Collect rollouts for a batch of episodes
        with torch.no_grad():
            while timesteps_collected < self.config.n_steps:
                ep_rewards, ep_values = [], []
                # Reset environment
                state, _ = self.env.reset()
                done = False
                # Collect data for a single episode
                for _ in range(self.config.max_timesteps):
                    if self.config.render:
                        self.env.render()
                    timesteps_collected += 1
                    # Collect state
                    states.append(state)
                    # Get action from the actor network
                    state_ = torch.tensor(state, dtype=torch.float, device=self.device)
                    action_ = self.actor.get_action(state_)
                    action = action_.item() if self.discrete else action_.cpu().numpy()
                    # Compute log probability of current policy
                    log_prob = self.actor.log_probs(state_, action_).item()
                    # Evaluate the current state with critic network
                    value = self.critic(state_)
                    # Take a step in the environment
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    # Store rewards, values, actions, and log_probs of current episode
                    ep_rewards.append(reward)
                    ep_values.append(value.item())
                    actions.append(action)
                    log_probs.append(log_prob)
                    if done:
                        break
                # Store rewards and values for all episodes
                rewards.append(ep_rewards)
                values.append(ep_values)

        # Convert to tensors for computation
        states = torch.tensor(np.array(states), dtype=torch.float, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long if self.discrete else torch.float, device=self.device)
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float, device=self.device)

        return states, actions, log_probs, rewards, values

    def _calculate_gae(self, rewards, values):
        """Compute advantages using Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        rewards : list
            List of length n_episodes containing rewards per episode.
        values : list
            List of length n_episodes containing value estimates per episode.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_steps,) containing advantages.
        """
        advantages = []
        for ep_rewards, ep_values in zip(rewards, values):
            ep_len = len(ep_rewards)
            ep_adv = torch.zeros(ep_len, dtype=torch.float, device=self.device)
            last_adv = 0.0
            for t in reversed(range(ep_len)):
                # Determine next value
                next_val = ep_values[t+1] if t < ep_len - 1 else 0.0
                # Compute temporal difference error
                delta = ep_rewards[t] + self.config.gamma * next_val - ep_values[t]
                # GAE advantage update
                last_adv = delta + self.config.gamma * self.config.lambd * last_adv
                ep_adv[t] = last_adv
            advantages.append(ep_adv)
        return torch.cat(advantages)


    def _update_networks(self, dataloader, total_timesteps, curr_timestep):
        """Update Actor and Critic networks with PPO-clip loss.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader object for iterating over mini-batches.
        total_timesteps : int
            Total number of simulation steps for training.
        curr_timestep : int
            Current timestep in the training process.

        Returns
        -------
        actor_losses : list
            Actor loss for each mini-batch.
        critic_losses : list
            Critic loss for each mini-batch.
        total_losses : list
            PPO objective loss for each mini-batch.
        """

        actor_losses, critic_losses, total_losses = [], [], []

        # Linearly decay learning rate
        lr = max(self.config.lr * (1 - (curr_timestep - 1) / total_timesteps), 0.0)
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr

        # Update actor and critic networks
        for _ in range(self.config.n_epochs):
            for data in dataloader:
                # Unpack mini-batch
                states, actions, log_probs, advantages, rewards_to_go = data
                # Move tensors to device for computation
                states = states.to(self.device)
                actions = actions.to(self.device)
                log_probs = log_probs.to(self.device)
                advantages = advantages.to(self.device)
                rewards_to_go = rewards_to_go.to(self.device)
                # Evaluate current states with critic network
                values_ = self.critic(states)
                # Compute log probabilities and entropy of current policy
                curr_log_probs = self.actor.log_probs(states, actions)
                entropy = self.actor.entropy(states)
                # Compute KL divergence
                logratios = curr_log_probs - log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()
                # Compute actor loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.config.clip, 1 + self.config.clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Entropy regularization
                entropy_term = self.config.ent_coef * entropy
                actor_loss -= entropy_term
                # Compute critic loss
                critic_loss = nn.MSELoss()(values_, rewards_to_go)
                # Compute total loss
                total_loss = actor_loss + self.config.vt_coef * critic_loss
                # Back-propagate actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()
                # Back-propagate critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()
                # Collect losses
                actor_losses.append((actor_loss + entropy_term).item())
                critic_losses.append(critic_loss.item())
                total_losses.append(total_loss.item())
                # Early stopping based on KL divergence
                if approx_kl.item() > self.config.target_kl:
                    break

        return actor_losses, critic_losses, total_losses

    def _log_metrics(self, csv_file, csv_writer, pbar, iteration, sim_timesteps,
                    batch_timesteps, avg_return, avg_actor_loss, avg_critic_loss, avg_total_loss):
        pbar.set_postfix({
            'iteration': iteration,
            'return': f"{avg_return:.2f}",
            'actor_loss': f"{avg_actor_loss:.5f}",
            'critic_loss': f"{avg_critic_loss:.5f}",
            'total_loss': f"{avg_total_loss:.5f}",
            'lr': f"{self.actor_optim.param_groups[0]['lr']:.5f}"
        })
        pbar.update(batch_timesteps)
        csv_writer.writerow([iteration, sim_timesteps, avg_return, avg_actor_loss, avg_critic_loss, avg_total_loss])
        csv_file.flush()

    def _save_models(self):
        torch.save(self.actor.state_dict(), f'saved/models/{self.config.env_name}/ppo/ppo_actor_{self.config.seed}.pth')
        torch.save(self.critic.state_dict(), f'saved/models/{self.config.env_name}/ppo/ppo_critic_{self.config.seed}.pth')


class DEPPO(PPO):

    def _compute_entropy_coefficient(self, t, total_timesteps):
        """Apply a sinusoidal oscillation to the entropy coefficient.

        Parameters
        ----------
        t : int
            Current timestep in the training process.
        total_timesteps : int
            Total number of simulation steps for training.

        Returns
        -------
        float
            Entropy coefficient for the current timestep.
        """
        phase_shift = self.config.base_phase + self.config.phase_slope * (t / total_timesteps)
        amplitude = self.config.ent_coef_amp * np.exp(-self.config.lambda_decay * t / total_timesteps)
        # Sine oscillation
        oscillation = np.sin(2 * np.pi * self.config.ent_coef_freq * t / total_timesteps + phase_shift)
        # Adjusted entropy coefficient
        ent_coef = (self.config.ent_coef_base + amplitude * oscillation)
        return ent_coef
    

    def _update_networks(self, dataloader, total_timesteps, curr_timestep):
        actor_losses, critic_losses, total_losses = [], [], []

        # Linearly decay learning rate
        lr = max(self.config.lr * (1 - (curr_timestep - 1) / total_timesteps), 0.0)
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr

        # Update actor and critic networks
        for _ in range(self.config.n_epochs):
            for data in dataloader:
                # Unpack mini-batch
                states, actions, log_probs, advantages, rewards_to_go = data
                # Move tensors to device for computation
                states = states.to(self.device)
                actions = actions.to(self.device)
                log_probs = log_probs.to(self.device)
                advantages = advantages.to(self.device)
                rewards_to_go = rewards_to_go.to(self.device)
                # Evaluate current states with critic network
                values_ = self.critic(states)
                # Compute log probabilities and entropy of current policy
                curr_log_probs = self.actor.log_probs(states, actions)
                entropy = self.actor.entropy(states)
                # Compute KL divergence
                logratios = curr_log_probs - log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()
                # Compute actor loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(
                        ratios, 1 - self.config.clip, 1 + self.config.clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Entropy regularization
                ent_coef = self._compute_entropy_coefficient(curr_timestep, total_timesteps)
                entropy_term = ent_coef * entropy
                actor_loss -= entropy_term
                # Compute critic loss
                critic_loss = nn.MSELoss()(values_, rewards_to_go)
                # Compute total loss
                total_loss = actor_loss + self.config.vt_coef * critic_loss
                # Back-propagate actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()
                # Back-propagate critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()
                # Collect losses
                actor_losses.append((actor_loss + entropy_term).item())
                critic_losses.append(critic_loss.item())
                total_losses.append(total_loss.item())
                # Early stopping based on KL divergence
                if approx_kl.item() > self.config.target_kl:
                    break

        return actor_losses, critic_losses, total_losses

    def learn(self, total_timesteps, log_filename='de_ppo_logs'):
        return super().learn(total_timesteps, log_filename)
    
    def _save_models(self):
        torch.save(self.actor.state_dict(), f'saved/models/{self.config.env_name}/de-ppo/de_ppo_actor_{self.config.seed}.pth')
        torch.save(self.critic.state_dict(), f'saved/models/{self.config.env_name}/de-ppo/de_ppo_critic_{self.config.seed}.pth')