import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.reward_contributions = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.reward_contributions[:]

class ActorCritic_Clip(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, num_layers=1, action_std_init=0.6):
        super(ActorCritic_Clip, self).__init__()
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.actor_fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.actor_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.actor_fc4 = nn.Linear(hidden_size, action_dim)
        self.actor_activation = nn.Tanh()

        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.critic_fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.critic_fc4 = nn.Linear(hidden_size, 1)

    def forward(self):
        raise NotImplementedError

    def act(self, state, hidden_actor=None):
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if hidden_actor is None:
            # Ensure hidden_actor is properly initialized
            hidden_actor = (
                torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device),
            )

        x = torch.relu(self.actor_fc1(state))
        x = torch.relu(self.actor_fc2(x))
        x = torch.relu(self.actor_fc3(x))

        x = x.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, 128)
        x, hidden_actor = self.actor_lstm(x, hidden_actor)
        x = x.squeeze(0).squeeze(0)

        action_mean = self.actor_activation(self.actor_fc4(x))
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), hidden_actor

    def evaluate(self, state, action, hidden_critic=None):
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if hidden_critic is None:
            hidden_critic = (
                torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device),
            )

        x = torch.relu(self.critic_fc1(state))
        x = torch.relu(self.critic_fc2(x))
        x = torch.relu(self.critic_fc3(x))
        x = x.unsqueeze(1)
        x, hidden_critic = self.critic_lstm(x, hidden_critic)
        x = x.squeeze(1)

        state_value = self.critic_fc4(x)

        action_mean = self.actor_activation(self.actor_fc4(x))
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_value, dist_entropy, hidden_critic

    def evaluate_critic(self, state, hidden_critic = None):
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        hidden_critic = (
            torch.zeros(1, batch_size, 128).to(device),
            torch.zeros(1, batch_size, 128).to(device)
        )
        x = torch.relu(self.critic_fc1(state))
        x = torch.relu(self.critic_fc2(x))
        x = torch.relu(self.critic_fc3(x))

        x = x.unsqueeze(0).unsqueeze(1)
        x, hidden_critic = self.critic_lstm(x, hidden_critic)
        x = x.squeeze(0).squeeze(0)

        state_value = self.critic_fc4(x)
        return state_value

    def compute_loss(self, old_logprobs, logprobs, state_values, rewards, old_values, dist_entropy, advantages, eps_clip):
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)  # Added normalization
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * nn.MSELoss()(state_values, rewards)
        entropy_bonus = -0.01 * dist_entropy.mean()
        total_loss = policy_loss + value_loss + entropy_bonus
        return total_loss, policy_loss, value_loss, dist_entropy.mean()


class PPO_Clip:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic_Clip(state_dim, action_dim, action_std_init=action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.policy_old = ActorCritic_Clip(state_dim, action_dim, action_std_init=action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.hidden_actor = None
        self.hidden_critic = None

    def reset_hidden_states(self):
        # Reset hidden states for LSTM
        self.hidden_actor = (
            torch.zeros(1, 1, 128).to(device),
            torch.zeros(1, 1, 128).to(device),
        )
        self.hidden_critic = (
            torch.zeros(1, 1, 128).to(device),
            torch.zeros(1, 1, 128).to(device),
        )

    def select_action(self, state):
        with torch.no_grad():
            state_values = torch.FloatTensor(state).to(device)

            action, action_logprob, self.hidden_actor = self.policy_old.act(state_values, self.hidden_actor)
            state_value = self.policy.evaluate_critic(state_values, self.hidden_critic)
            self.buffer.states.append(state_values)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_value)

        return action.detach().cpu().numpy()

    ### Change !!!
    def update(self):
        rewards = []
        discounted_reward = 0
        reward_contributions = {
            "distance_reward": 0,
            "fall_penalty": 0,
            "symmetry_reward": 0,
            "period_penalty": 0,
            "contact_penalty": 0,
            "smoothness_penalty": 0,
            "progress_reward":0,
            "mass_centre_reward":0,
            "stability_penalty":0,
        }
        
        for reward, is_terminal, contributions in zip(reversed(self.buffer.rewards),
                                                    reversed(self.buffer.is_terminals),
                                                    reversed(self.buffer.reward_contributions)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

            for key in reward_contributions:
                reward_contributions[key] += contributions[key]

        # for key, value in reward_contributions.items():
        #     self.writer.add_scalar(f"Reward Contributions/{key}", value, self.current_episode)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    # def update(self):
    #     rewards = []
    #     discounted_reward = 0
    #     for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
    #         if is_terminal:
    #             discounted_reward = 0
    #         discounted_reward = reward + (self.gamma * discounted_reward)
    #         rewards.insert(0, discounted_reward)

    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # Normalize rewards
        rewards = torch.clamp(rewards, min=-10, max=10)  # Limit reward range

        old_states = torch.stack(self.buffer.states).detach().to(device)
        old_actions = torch.stack(self.buffer.actions).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values).detach().to(device)

        advantages = rewards - old_state_values

        total_loss, total_policy_loss, total_value_loss, total_entropy = 0, 0, 0, 0

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy, _ = self.policy.evaluate(
                old_states, old_actions, hidden_critic=self.hidden_critic
            )

            loss, policy_loss, value_loss, entropy = self.policy.compute_loss(
                old_logprobs=old_logprobs,
                logprobs=logprobs,
                state_values=state_values,
                rewards=rewards,
                old_values=old_state_values,
                dist_entropy=dist_entropy,
                advantages=advantages,
                eps_clip=self.eps_clip,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return (
            total_loss / self.K_epochs,
            total_policy_loss / self.K_epochs,
            total_value_loss / self.K_epochs,
            total_entropy / self.K_epochs,
        )

    # Change
    # def save(self, checkpoint_path):
    #     # Save optimizer state as well
    #     torch.save(
    #         {
    #             "model_state_dict": self.policy.state_dict(),
    #             "optimizer_state_dict": self.optimizer.state_dict(),
    #         },
    #         checkpoint_path,
    #     )

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

    # def load(self, checkpoint_path):
    #     # Load model weights only
    #     checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True)
    #     self.policy.load_state_dict(checkpoint)


    # def load(self, checkpoint_path):
    #     self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    #     self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
def process_state(state):
    robot_position = state['Robot Position']
    robot_orientation = state['Robot Orientation']
    robot_linear_velocity = state['Robot Linear Velocity']
    robot_angular_velocity = state['Robot Angular Velocity']
    joint_positions = state['Joint positions'].flatten()
    joint_velocities = state['Joint velocities'].flatten()
    state_vector = np.concatenate([robot_position, robot_orientation, robot_linear_velocity, robot_angular_velocity, joint_positions, joint_velocities])
    return state_vector

