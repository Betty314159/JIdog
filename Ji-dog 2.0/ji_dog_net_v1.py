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

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, 128)
        self.actor_fc2 = nn.Linear(128, 256)
        self.actor_fc3 = nn.Linear(256, 128)
        self.actor_lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.actor_fc4 = nn.Linear(128, action_dim)
        self.actor_activation = nn.Tanh()

        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, 128)
        self.critic_fc2 = nn.Linear(128, 256)
        self.critic_fc3 = nn.Linear(256, 128)
        self.critic_lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.critic_fc4 = nn.Linear(128, 1)

    def forward(self):
        raise NotImplementedError
        
    def act(self, state, hidden_actor=None):
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if hidden_actor is None:
            hidden_actor = (
                torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device)
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
        # Dynamically initialize hidden state for critic based on batch size
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if hidden_critic is None:
            hidden_critic = (
                torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device)
            )

        x = torch.relu(self.critic_fc1(state))
        x = torch.relu(self.critic_fc2(x))
        x = torch.relu(self.critic_fc3(x))
        # Adjust the dimensions to be compatible with LSTM
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, 128)
        x, hidden_critic = self.critic_lstm(x, hidden_critic)
        x = x.squeeze(1)  # Remove the sequence dimension

        state_value = self.critic_fc4(x)
        
        # Actor part for logprobs and entropy calculation
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
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2)
        value_loss = 0.5 * nn.MSELoss()(state_values, rewards)
        entropy_bonus = -0.01 * dist_entropy
        total_loss = policy_loss + value_loss + entropy_bonus
        return total_loss

class PPO_Clip:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.hidden_actor = None
        self.hidden_critic = None

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
    

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards.view(-1, 1)

        old_states = torch.stack(self.buffer.states).detach().to(device)
        old_actions = torch.stack(self.buffer.actions).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values).detach().to(device)

        advantages = rewards - old_state_values

        for _ in range(self.K_epochs):
            # Detach hidden states for each batch
            batch_size = old_states.size(0)
            self.hidden_critic = (
                torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device)
            )

            logprobs, state_values, dist_entropy, _ = self.policy.evaluate(
                old_states, old_actions, hidden_critic=self.hidden_critic
            )

            # Compute loss
            loss = self.policy.compute_loss(
                old_logprobs=old_logprobs,
                logprobs=logprobs,
                state_values=state_values,
                rewards=rewards,
                old_values=old_state_values,
                dist_entropy=dist_entropy,
                advantages=advantages,
                eps_clip=self.eps_clip
            )

            self.optimizer.zero_grad()
            loss = loss.mean()  # Ensure loss is a scalar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
def process_state(state):
    robot_position = state['Robot Position']
    robot_orientation = state['Robot Orientation']
    robot_linear_velocity = state['Robot Linear Velocity']
    robot_angular_velocity = state['Robot Angular Velocity']
    joint_positions = state['Joint positions'].flatten()
    joint_velocities = state['Joint velocities'].flatten()
    state_vector = np.concatenate([robot_position, robot_orientation, robot_linear_velocity, robot_angular_velocity, joint_positions, joint_velocities])
    return state_vector

