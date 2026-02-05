from pettingzoo.utils.wrappers import BaseParallelWrapper

class RewardSharingWrapper(BaseParallelWrapper):
    """
    A wrapper to modify the reward structure of the predators.
    
    Parameters:
    - env: The PettingZoo parallel environment.
    - alpha (float): The cooperation factor (0.0 to 1.0).
        * alpha = 1.0: Pure Individual Reward (Competition).
        * alpha = 0.0: Pure Shared Reward (Full Cooperation).
        * alpha = 0.5: Mixed Strategy.
    """
    def __init__(self, env, alpha=1.0):
        super().__init__(env)
        self.alpha = alpha
        # Identify which agents are predators (adversaries)
        self.group_agents = [a for a in env.possible_agents if "adversary" in a]

    def step(self, actions):
        # Perform the standard environment step
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Calculate the total reward achieved by the predator group
        group_reward_sum = sum(rewards[a] for a in self.group_agents)
        
        # Redistribute rewards based on alpha
        new_rewards = rewards.copy()
        
        for agent in self.group_agents:
            individual_r = rewards[agent]
            
            # Formula: (alpha * Own_Reward) + ((1-alpha) * Group_Total)
            shared_r = (self.alpha * individual_r) + ((1 - self.alpha) * group_reward_sum)
            new_rewards[agent] = shared_r
            
        return obs, new_rewards, terminations, truncations, infos