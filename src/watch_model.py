import time
from stable_baselines3 import PPO, A2C
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.snake_env import SnakeTestEnv

# === choose your trained model ===
MODEL_PATH = "models/ppo_snake_collector_seed7_20251023_000447.zip"

env = SnakeTestEnv(reward_mode="collector", seed=7)
model = PPO.load(MODEL_PATH)

obs, info = env.reset()
fps = 24
dt = 1.0 / fps

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    print("\033[H\033[J")  # clear screen
    print(env.render())
    print(
        f"Reward={reward:.2f} | Steps={info['steps']} | Apples={info['apples']} | "
        f"Coverage={info['coverage_ratio']*100:.1f}% | Died={info['died']}"
    )
    time.sleep(dt)
    if terminated or truncated:
        obs, info = env.reset()
