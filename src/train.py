
import argparse
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from envs.snake_env import SnakeTestEnv
from envs.webflow_env import WebFlowEnv
from envs.mock_webflow_env import MockWebFlowEnv

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
}

def make_vec_env(env_name: str, seed: int, n_envs: int):
    if n_envs <= 1:
        return DummyVecEnv([make_env(env_name, seed)])
    def thunk(i):
        return lambda: make_env(env_name, seed + i)()
    return SubprocVecEnv([thunk(i) for i in range(n_envs)])

def make_env(env_name: str, seed: int):
    # Map simple names to env configs
    if env_name == "snake_collector":
        return lambda: SnakeTestEnv(reward_mode="collector", seed=seed)
    if env_name == "snake_explorer":
        return lambda: SnakeTestEnv(reward_mode="explorer", seed=seed)
    if env_name == "snake_bug_hunter":
        return lambda: SnakeTestEnv(reward_mode="bug_hunter", seed=seed, fault_self_collision_skip=True, self_collision_skip_prob=0.3)
    if env_name == "snake_fault_invisible":
        return lambda: SnakeTestEnv(reward_mode="collector", seed=seed, fault_invisible_wall=True)
    if env_name == "snake_fault_delayed":
        return lambda: SnakeTestEnv(reward_mode="collector", seed=seed, fault_delayed_food_steps=12)
    if env_name == "web_completer":
        return lambda: WebFlowEnv(reward_mode="completer", step_limit=120, headless=True)
    if env_name == "web_fuzzer":
        return lambda: WebFlowEnv(reward_mode="fuzzer", step_limit=150, headless=True)
    if env_name == "web_bug_hunter":
        return lambda: WebFlowEnv(reward_mode="bug_hunter", step_limit=150, headless=True)
    if env_name == "web_mock_completer":
        return lambda: MockWebFlowEnv(reward_mode="completer", step_limit=80, curriculum=True)
    if env_name == "web_mock_fuzzer":
        return lambda: MockWebFlowEnv(reward_mode="fuzzer", step_limit=100, curriculum=True)
    if env_name == "web_mock_bug_hunter":
        return lambda: MockWebFlowEnv(reward_mode="bug_hunter", step_limit=80)

    raise ValueError(f"Unknown env_name: {env_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","a2c"], default="ppo")
    parser.add_argument("--env", default="snake_survivor")
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--n_envs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    env_fn = make_env(args.env, args.seed)
    vec_env = make_vec_env(args.env, args.seed, args.n_envs)

    Algo = ALGOS[args.algo]
    run_name = f"{args.algo}_{args.env}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(args.models_dir, run_name)

    if args.algo == "a2c":
        model = A2C(
            "MlpPolicy",
            vec_env,
            n_steps=128,        # ↑ default is 5; this matters a lot
            ent_coef=0.01,      # encourage exploration
            learning_rate=7e-4, # fine for A2C
            gamma=0.99,
            vf_coef=0.5,
            seed=args.seed,
            tensorboard_log=args.logdir,
            verbose=1,
            device="cpu"
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            tensorboard_log=args.logdir,
            seed=args.seed,
            verbose=1,
            device="cpu"
        )
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=args.models_dir, name_prefix=run_name)
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
