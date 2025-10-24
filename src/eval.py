
import argparse
import csv
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.snake_env import SnakeTestEnv
from envs.webflow_env import WebFlowEnv

ALGOS = {"ppo": PPO, "a2c": A2C}

def make_env(env_name: str, seed: int):
    if env_name == "snake_collector":
        return lambda: SnakeTestEnv(reward_mode="collector", seed=seed)
    if env_name == "snake_explorer":
        return lambda: SnakeTestEnv(reward_mode="explorer", seed=seed)
    if env_name == "snake_collector_invisible":
        return lambda: SnakeTestEnv(reward_mode="collector", seed=seed, fault_invisible_wall=True)
    if env_name == "snake_explorer_invisible":
        return lambda: SnakeTestEnv(reward_mode="explorer", seed=seed, fault_invisible_wall=True)
    if env_name == "web_completer":
        return lambda: WebFlowEnv(reward_mode="completer", step_limit=120, headless=True, curriculum=True)
    if env_name == "web_fuzzer":
        return lambda: WebFlowEnv(reward_mode="fuzzer", step_limit=150, headless=True)
    if env_name == "web_bug_hunter":
        return lambda: WebFlowEnv(reward_mode="bug_hunter", step_limit=80, headless=True)
    raise ValueError(f"Unknown env_name: {env_name}")

# def rollout(model, env, episodes=20):
#     rows = []
#     for ep in range(episodes):
#         obs, info = env.reset()
#         done = False
#         trunc = False
#         ep_rew = 0.0
#         while not (done or trunc):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, trunc, info = env.step(int(action))
#             ep_rew += float(reward)
#         row = {
#             "episode": ep,
#             "reward": ep_rew,
#             **info,
#         }
#         rows.append(row)
#     return rows

def rollout(model, env, episodes=20, trace_first=True):
    rows = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        ep_rew = 0.0
        steps = 0
        if trace_first and ep == 0:
            print("\n--- TRACE: episode 0 ---")
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            ep_rew += float(reward)
            steps += 1
            if trace_first and ep == 0:
                # infer page from info flags
                page = "home" if info.get("page_home") else ("signup1" if info.get("page_signup1")
                        else ("signup2" if info.get("page_signup2") else ("confirm" if info.get("page_confirm") else "?")))
                print(f"t={steps:02d} page={page:8s} action={int(action)} reward={reward:+.3f} "
                      f"err={info.get('error_banner',0)} clicked={info.get('clicked_unique',0)}")
        row = {"episode": ep, "reward": ep_rew, **info}
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","a2c"], required=True)
    parser.add_argument("--env", default="snake_collector")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--csv_out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = make_env(args.env, args.seed)()
    Algo = ALGOS[args.algo]
    model = Algo.load(args.model_path, device="cpu")

    rows = rollout(model, env, episodes=args.episodes)

    # print aggregates
    import statistics as stats
    r = [row["reward"] for row in rows]

    # detect environment type dynamically
    sample = rows[0] if rows else {}
    if "died" in sample:  # Snake env
        died = [row["died"] for row in rows]
        apples = [row["apples"] for row in rows]
        cov = [row["coverage_ratio"] for row in rows]
        print(f"Episodes: {len(rows)}")
        print(f"Reward: mean={np.mean(r):.2f}, median={np.median(r):.2f}")
        print(f"Death rate: {np.mean(died):.2%}")
        print(f"Apples/ep: mean={np.mean(apples):.2f}")
        print(f"Coverage: mean={np.mean(cov):.2%}")

    else:  # Web env
        succ = [row.get("success", 0) for row in rows]
        errs = [row.get("validation_errors", 0) for row in rows]
        print(f"Episodes: {len(rows)}")
        print(f"Reward: mean={np.mean(r):.2f}, median={np.median(r):.2f}")
        print(f"Success rate: {np.mean(succ):.2%}")
        print(f"Validation errors/ep: mean={np.mean(errs):.2f}")
    if args.csv_out:
        fieldnames = list(rows[0].keys()) if rows else []
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv_out}")

if __name__ == "__main__":
    main()
