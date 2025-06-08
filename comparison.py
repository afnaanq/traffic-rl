# compare_dqn_ddqn.py

from envs.intersection_env import CityFlowEnv
from train_dqn import train_dqn
from train_ddqn import train_ddqn

config_path = "./mydata/config.json"
intersection_id = "intersection_1_1"

# Reward function (choose one or make a loop over options)
reward_fn = lambda env: env.default_reward()

# === Set up environments (separate instances to avoid shared state) ===
env_dqn = CityFlowEnv(config_path, [intersection_id])
env_dqn.set_rwd_fn(lambda: reward_fn(env_dqn))

env_ddqn = CityFlowEnv(config_path, [intersection_id])
env_ddqn.set_rwd_fn(lambda: reward_fn(env_ddqn))

# === Train DQN ===
#print("\n===== Training DQN =====")
#train_dqn(env_dqn)

# === Train Double DQN ===
print("\n===== Training Double DQN =====")
train_ddqn(env_ddqn)
