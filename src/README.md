
# DRL for Automated Testing — Snake + Web (from scratch)

This repo gives you a **DRL testing framework** with a **Snake** game tailored for *testing*,
not just score maxing. It supports **personas** (collector / explorer), **fault injection**, rich **metrics**,
and scripts to **train / evaluate**.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```


### Snake

#### Train

```bash
# PPO on collector persona
python src/train.py --algo ppo --env snake_collector --timesteps 1000000 --seed 7

# Explorer persona
python src/train.py --algo ppo --env snake_explorer --timesteps 1000000 --seed 7

# A2C variant
python src/train.py --algo a2c --env snake_collector --timesteps 2000000 --seed 7
```

TensorBoard:
```bash
tensorboard --logdir logs
```

### Evaluate (CSV)

```bash
python src/eval.py --algo ppo --env snake_collector \
  --model_path models/ppo_snake_collector_seed7_*.zip \
  --episodes 100 --csv_out logs/snake_collector_eval.csv
```

```bash
python src/eval.py --algo ppo --env snake_explorer \
  --model_path models/ppo_snake_explorer_seed7_*.zip \
  --episodes 100 --csv_out logs/snake_explorer_eval.csv
```

```bash
python src/eval.py --algo a2c --env snake_collector \
  --model_path models/ppo_snake_a2c_collector_seed7_*.zip \
  --episodes 100 --csv_out logs/snake_a2c_collector_eval.csv
```

```bash
python src/eval.py --algo ppo --env snake_collector_invisible \
  --model_path models/ppo_snake_collector_seed7_*.zip \
  --episodes 100 --csv_out logs/snake_collector_invisible_eval.csv
```

```bash
python src/eval.py --algo ppo --env snake_explorer_invisible \
  --model_path models/ppo_snake_explorer_seed7_*.zip \
  --episodes 100 --csv_out logs/snake_explorer_invisible_eval.csv
```

CSV columns include `steps, apples, length, died, cause, wall_hits, self_hits, turns, unique_cells, coverage_ratio, time_since_last_food, reward`.

## Personas

- **collector**: +10 apple, −10 death, −0.01/step (time pressure)
- **explorer**: +coverage gain each step, +2 apple, −10 death

## Fault toggle

- **invisible wall** (one hidden blocking cell) → tests pathing/coverage


Use `--env snake_fault_invisible` or `--env snake_fault_delayed` to train/eval with specific bugs.

## Web Flow (Flask + Playwright)

This repo also includes a simple two-step signup app plus DRL personas that interact with it through Playwright. You can train/eval the agents on a mock environment for speed, then run evaluations against the real Flask app with optional fault injection (delay, HTTP 500).

### Setup

Start the web app (choose one depending on your repo layout):

```bash
$env:FAULT_DELAY_SEC=0
python webapp.py      # or: python app.py
```

Train (mock, fast)
```bash
python src/train.py --algo ppo --env web_mock_completer --timesteps 150000 --seed 7
```
```bash
python src/train.py --algo ppo --env web_mock_fuzzer  --timesteps 150000 --seed 7
```

### Evaluate (CSV)

Baseline (no faults) — real app
```bash
$env:FAULT_DELAY_SEC=0
```
# Completer
```bash
$m = Get-ChildItem models\ppo_web_mock_completer_seed7_*.zip | Sort-Object LastWriteTime | Select-Object -Last 1
python src\eval.py --algo ppo --env web_completer  --model_path "$($m.FullName)" --episodes 50 --csv_out logs\web_completer_eval.csv
```

Delay Fault (latency) — Completer under delay
```bash
$env:FAULT_DELAY_SEC=0.3   # use 0.3s to avoid Playwright 500ms timeout
$env:FAULT_EMAIL_500=0
$m = Get-ChildItem models\ppo_web_mock_completer_seed7_*.zip | Sort-Object LastWriteTime | Select-Object -Last 1
python src\eval.py --algo ppo --env web_completer --model_path "$($m.FullName)" --episodes 50 --csv_out logs\web_completer_delay_eval.csv
```



Fuzzer — real app
```bash
$env:FAULT_DELAY_SEC=0
$env:FAULT_EMAIL_500=0
$m = Get-ChildItem models\ppo_web_mock_fuzzer_seed7_*.zip | Sort-Object LastWriteTime | Select-Object -Last 1
python src\eval.py --algo ppo --env web_fuzzer --model_path "$($m.FullName)" --episodes 30 --csv_out logs\web_fuzzer_eval.csv
```

## Reproducibility

- Seeds on env and algo (`--seed`)
- Saved models in `models/`
- TensorBoard logs in `logs/`
- Exact CLI commands documented above

