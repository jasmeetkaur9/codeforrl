import torch
import gymnasium as gym


class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = input_dim 
        self.output = output_dim

        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output)
        )

    def sample(self, x):
        logits = self.network(x)
        distribution = torch.distributions.Categorical(logits = logits)
        action = distribution.sample()
        pb = torch.exp(distribution.log_prob(action))
        return action.item(), pb

    def get_prob(self, x, action):
        logits = self.network(x)
        distribution = torch.distributions.Categorical(logits = logits)
        pb = torch.exp(distribution.log_prob(action))
        return pb 
    
    def forward(self, x):
        logits = self.network(x)
        return logits

class Valuefunction(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = input_dim 

        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class PPO:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.policy_old = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.policy_new = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.policy_optimizer = torch.optim.Adam(self.policy_new.parameters(), lr = 0.0001)
        self.valuefunction = Valuefunction(self.env.observation_space.shape[0])
        self.value_optimizer = torch.optim.Adam(self.valuefunction.parameters(), lr = 0.001)


    def train(self, episodes):

        moving_rt = []
        moving_el = []
        batch_p = []
        batch_v = []
        epsilon = 0.2
        update_freq = 5

        for ep in range(0, episodes):
            obs, _ = self.env.reset()
            done = False
            gamma = 0.99
            rt = 0
            traj_advantage = 0
            traj_loss = 0
            t = 0
            while not done :
                action, prob_old = self.policy_old.sample(torch.tensor(obs))
                next_obs, reward, done, _ , _ = self.env.step(action)

                # collect trajectory data
                rt = rt + pow(gamma, t)*reward
                
                value = self.valuefunction(torch.tensor(obs))
                advantage = rt 

                # traj_loss = traj_loss + torch.square(advantage)
                
                prob_new = self.policy_new.get_prob(torch.tensor(obs), torch.tensor(action))
                ratio = prob_new/prob_old
                av1 = ratio * advantage
                clipped_ratio = torch.clamp((prob_new/prob_old), min=1-epsilon, max=1+epsilon)
                av2 = clipped_ratio * advantage
                advantage = torch.min(av1, av2)

                traj_advantage = traj_advantage + advantage
                t = t+1
                obs = next_obs

            batch_p.append(traj_advantage)
            # batch_v.append(traj_loss)
            moving_rt.append(rt)
            moving_el.append(float(t))

            if ep%5 == 0:
                loss1 = -torch.mean(torch.tensor(batch_p, requires_grad = True))
                self.policy_optimizer.zero_grad()
                loss1.backward()
                self.policy_optimizer.step()
                if ep%update_freq == 0:
                    for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
                            old_param.data.copy_(new_param.data)

            # loss2 = torch.mean(torch.tensor(batch_v, requires_grad = True))
            # self.value_optimizer.zero_grad()
            # loss2.backward()
            # self.value_optimizer.step()

            if ep%50 == 0:
                print("Return :", torch.mean(torch.tensor(moving_rt)))

           


if __name__ == '__main__':

    agent = PPO()
    agent.train(10000)


            

                

                

                




    


