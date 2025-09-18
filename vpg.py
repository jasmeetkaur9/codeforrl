import torch
import torch.distributions as td
import gymnasium as gym


class Policy(torch.nn.Module):
    def __init__(self, obs_dim, n_action):
        super().__init__()
        self.obs_dim = obs_dim 
        self.output = n_action 

        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output)
        )

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = 0.001)

    
    def sample(self, obs):
        with torch.no_grad():
            probs = self.policy_network(torch.tensor(obs))
            pdf = td.Categorical(logits=probs)
            return pdf.sample().item()

    def forward(self, obs):
        return self.policy_network(obs)

    
class VPG:

    def __init__(self):

        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.obs_dim = self.env.observation_space.shape[0] 
        self.n_action = self.env.action_space.n
        self.policy = Policy(self.obs_dim, self.n_action)

    
    def train(self, episodes):

        obs, _  = self.env.reset()


        for ep in range(episodes):

            done = False
            obs, _ = self.env.reset()
            tr_reward = []
            actions = []
            states = []

            while not done:
                action = self.policy.sample(obs)
                next_obs, reward, done, _, _ = self.env.step(action)
                tr_reward.append(reward)
                states.append(obs)
                actions.append(action)
                obs = next_obs

            
            tr_reward = torch.tensor(tr_reward)
            states = torch.tensor(states)
            actions = torch.tensor(actions)

            log_probs = td.Categorical(logits=self.policy.forward(states)).log_prob(actions)
            log_probs = torch.sum(log_probs)
            loss = -log_probs*torch.sum(tr_reward)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            if ep%50 == 0 :
                print("Episode Reward :", torch.sum(tr_reward))



if __name__ == '__main__':

    vpg = VPG()
    vpg.train(1000)
    
        







