from stable_baselines import A2C, PPO2, TRPO, PPO1, ACKTR
from stable_baselines.common.evaluation import evaluate_policy
import seed_generation
from one_dimensional_env import BikesEnv
from simple_2_dimensional_env import TwoDBikesEnv


# Create environment
print("env")
env = TwoDBikesEnv(12, "ACKTR-2d")

# Instantiate the agent
print("model")
model = ACKTR('MlpPolicy', env, verbose=1)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)
# Train the agent
print("learn")
timesteps = int(2e5)
model.learn(total_timesteps=timesteps)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)
