# Survival Dynamics of Neural and Programmatic Policies in an Evolutionary Reinforcement Learning Environment

An open-source reimplementation of the classic Ackley & Littman (1991) Artificial Life (ALife) predator–prey testbed, designed for studying evolutionary reinforcement learning in reward-sparse, partially observable environments.

This repo:

- Introduces **Programmatic ERL (PERL)** policies—soft, differentiable decision lists evolved and trained in a sparse-survival setting.
- Rebuilds and modernizes the original **Ackley & Littman (1991)** ALife ecosystem with a clean Python API and reproducible experiments.
- Provides a full **survival analysis pipeline** (Kaplan–Meier, RMST, log-rank tests) to quantitatively compare neural ERL (NERL) and PERL on long-horizon survival.

---

Developed by **Anton Roupassov-Ruiz** and **Willy Zuo** during  
*CMPUT 497/651 – Program Synthesis, Heuristic Search and Artificial Life*  
at the **University of Alberta** (Fall 2025), under the supervision of **Dr. Vadim Bulitko**.

### Dependencies

This project is implemented in Python and relies on the following libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `lifelines`  <!-- survival analysis -->
- `math` (from the Python standard library)

### Quick Start

To run the experiment with default settings:

```bash
python3 run_quick_experiment.py
```

You can adjust the number of trials and maximum steps by modifying the constants in `run_quick_experiment.py`:

```python
TRIALS_PER_STRATEGY = 500  # Number of independent trials per strategy
MAX_STEPS = 2000           # Maximum duration of each trial
```

#### Key Files

- **`run_quick_experiment.py`**: Orchestrates the comparative experiments. It runs multiple parallel trials for different strategies (e.g., ERL, Programmatic, Evolution-only), collects survival data, and performs statistical analysis (Kaplan-Meier, Log-Rank, RMST) to generate plots and significance tests.
- **`programmatic_erl.py`**: Implements the **Programmatic ERL (PERL)** agent. This agent uses a structured, interpretable genome consisting of soft decision lists (clauses) for both action selection and state evaluation. It supports evolution (mutation/crossover) and lifetime learning (policy gradient on clauses).
- **`ERL.py`**: Implements the **Neural ERL (NERL)** agent, faithfully reproducing the original Ackley & Littman (1991) specification. The agent uses a feed-forward neural network for its policy and evaluation functions. This file also contains the core simulation logic (World, Entities) shared by both agent types.

#### Live Simulation

To watch a live visualization of the gridworld simulation (using the Neural ERL agent by default):

```bash
python3 ERL.py
```
