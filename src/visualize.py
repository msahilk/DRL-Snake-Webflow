import argparse
import time
from stable_baselines3 import PPO, A2C   # 👈 import algorithms
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.snake_env import SnakeTestEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="snake_collector")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")   # 👈 choose which algorithm
    parser.add_argument("--model_path", required=True)                     # 👈 pass your trained model path
    args = parser.parse_args()

    # Pick environment setup
    if args.env == "snake_collector":
        env = SnakeTestEnv(reward_mode="collector", seed=args.seed)
    elif args.env == "snake_explorer":
        env = SnakeTestEnv(reward_mode="explorer", seed=args.seed)
    elif args.env == "snake_bug_hunter":
        env = SnakeTestEnv(reward_mode="bug_hunter", seed=args.seed, fault_self_collision_skip=True)
    elif args.env == "snake_fault_invisible":
        env = SnakeTestEnv(reward_mode="collector", seed=args.seed, fault_invisible_wall=True)
    elif args.env == "snake_fault_delayed":
        env = SnakeTestEnv(reward_mode="collector", seed=args.seed, fault_delayed_food_steps=12)
    else:
        raise ValueError("Unknown env")

    # Load model 👇
    Algo = PPO if args.algo == "ppo" else A2C
    model = Algo.load(args.model_path)
    print(f"Loaded {args.algo.upper()} model from {args.model_path}")

    # Run loop
    obs, info = env.reset()
    dt = 1.0 / args.fps

    while True:
        action, _ = model.predict(obs, deterministic=True)  # 👈 use trained policy
        obs, reward, terminated, truncated, info = env.step(int(action))

        print("\033[H\033[J")  # clear screen
        print(env.render())
        print(
            f"Reward={reward:.3f} | Steps={info['steps']} | Apples={info['apples']} | "
            f"Coverage={info['coverage_ratio']*100:.1f}% | Died={info['died']} | Cause={info['cause']}"
        )

        time.sleep(dt)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
