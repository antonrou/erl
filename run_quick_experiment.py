import ERL
import programmatic_erl
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import pandas as pd
from lifelines import CoxPHFitter

# Configuration for Quick Verification
STRATEGIES = ['ERL', 'E', 'L', 'F', 'Programmatic', 'PE', 'PL', 'B']
TRIALS_PER_STRATEGY = 100 # Reduced for quick baseline
MAX_STEPS = 2000

# Strategy display names for plot legends
STRATEGY_LABELS = {
    'E': 'NE',
    'L': 'NL',
    'F': 'NF',
    'Programmatic': 'PERL',
    'ERL': 'NERL',
    'PE': 'PE',
    'PL': 'PL',
    'B': 'B'
}

def run_single_trial(strategy, trial_num, seed_offset):
    # Use trial_num + offset as seed for reproducibility within a run, but variance across runs
    current_seed = trial_num + seed_offset
    # print(f"[{strategy}] Starting Trial {trial_num+1}/{TRIALS_PER_STRATEGY} (Seed: {current_seed})")
    if strategy in ['Programmatic', 'PE', 'PL']:
        steps, history = programmatic_erl.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS, seed=current_seed)
    else:
        steps, history = ERL.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS, seed=current_seed)
    return steps, current_seed, history

def run_experiments(strategies=None):
    if strategies is None:
        strategies = STRATEGIES
    results = {s: [] for s in strategies}
    
    print(f"Running {TRIALS_PER_STRATEGY} trials (max {MAX_STEPS} steps) for each strategy using {multiprocessing.cpu_count()} CPU cores...")
    
    # Submit ALL tasks at once
    with multiprocessing.Pool() as pool:
        all_async_results = []
        
        # Generate a random seed offset for this run
        seed_offset = int(time.time()) % 10000
        print(f"Using seed offset: {seed_offset}")

        for strategy in strategies:
            for i in range(TRIALS_PER_STRATEGY):
                res = pool.apply_async(run_single_trial, (strategy, i, seed_offset))
                all_async_results.append((strategy, i, res))
        
        start_time = time.time()
        total_trials = len(all_async_results)
        finished_count = 0
        
        while finished_count < total_trials:
            for idx, (strategy, trial_num, res) in enumerate(all_async_results):
                if res is None: continue
                
                if res.ready():
                    try:
                        steps, seed, history = res.get()
                        results[strategy].append((steps, seed, history))
                        all_async_results[idx] = (strategy, trial_num, None)
                        finished_count += 1
                        # print(f"[{strategy}] Trial {trial_num+1} finished: {steps}")
                    except Exception as e:
                        print(f"[{strategy}] Trial {trial_num+1} failed: {e}")
                        all_async_results[idx] = (strategy, trial_num, None)
                        finished_count += 1
            
            time.sleep(0.5)
            
        duration = time.time() - start_time
        print(f"All experiments finished in {duration:.2f}s")
    
    # Analyze Results
    print("\n--- RESULTS (Average Steps Survived) ---")
    for s in strategies:
        steps_data = [r[0] for r in results[s]]
        avg = np.mean(steps_data) if steps_data else 0
        med = np.median(steps_data) if steps_data else 0
        survived_full = sum(1 for x in steps_data if x >= MAX_STEPS)
        print(f"{s}: Avg={avg:.1f}, Median={med:.1f}, Survived Full Duration={survived_full}/{TRIALS_PER_STRATEGY}")
    
    plot_kaplan_meier(results)
    perform_log_rank_test(results)
    perform_cox_ph_analysis(results)
    return results

def perform_cox_ph_analysis(results):
    """
    Perform Cox Proportional Hazards regression to compare Programmatic ERL vs Neural ERL.
    
    Outputs:
    - Hazard Ratio (HR)
    - 95% Confidence Interval
    - p-value
    """
    print("\n--- Cox Proportional Hazards Analysis ---")
    if 'ERL' not in results or 'Programmatic' not in results:
        print("Cannot perform Cox PH Analysis: Missing 'ERL' or 'Programmatic' data.")
        return
    
    # Prepare data
    times = []
    death_flags = []
    strategy_labels = []
    
    # Add ERL data
    for steps, seed, history in results['ERL']:
        times.append(steps)
        death_flags.append(1 if steps < MAX_STEPS else 0)  # 1 = death, 0 = censored
        strategy_labels.append("neural")
    
    # Add Programmatic data
    for steps, seed, history in results['Programmatic']:
        times.append(steps)
        death_flags.append(1 if steps < MAX_STEPS else 0)
        strategy_labels.append("programmatic")
    
    # Create DataFrame
    df = pd.DataFrame({
        "time": times,
        "event": death_flags,
        "strategy": strategy_labels,
    })
    
    # Create binary variable: 1 for programmatic, 0 for neural
    df["strategy_binary"] = (df["strategy"] == "programmatic").astype(int)
    
    # Drop the original string 'strategy' column
    df = df.drop(columns=['strategy'])
    
    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    
    # Extract results
    summary = cph.summary
    
    print("\nCox PH Model Summary:")
    print(f"{'Covariate':<20} {'Hazard Ratio':<15} {'95% CI':<25} {'p-value':<10}")
    print("-" * 70)
    
    for idx in summary.index:
        hr = summary.loc[idx, 'exp(coef)']
        ci_lower = summary.loc[idx, 'exp(coef) lower 95%']
        ci_upper = summary.loc[idx, 'exp(coef) upper 95%']
        p_val = summary.loc[idx, 'p']
        
        ci_str = f"({ci_lower:.3f}, {ci_upper:.3f})"
        print(f"{idx:<20} {hr:<15.3f} {ci_str:<25} {p_val:<10.4f}")
    
    print("\nInterpretation:")
    hr_prog = summary.loc['strategy_binary', 'exp(coef)']
    p_val = summary.loc['strategy_binary', 'p']
    
    if hr_prog > 1:
        risk_change = (hr_prog - 1) * 100
        print(f"Programmatic ERL has {risk_change:.1f}% higher hazard of death than Neural ERL (HR={hr_prog:.3f}).")
    else:
        risk_reduction = (1 - hr_prog) * 100
        print(f"Programmatic ERL has {risk_reduction:.1f}% lower hazard of death than Neural ERL (HR={hr_prog:.3f}).")
    
    if p_val < 0.05:
        print(f"This difference is statistically significant (p={p_val:.4f}).")
    else:
        print(f"This difference is not statistically significant (p={p_val:.4f}).")


def perform_log_rank_test(results):
    print("\n--- Log-Rank Test (Mantel-Cox) ---")
    if 'ERL' not in results or 'Programmatic' not in results:
        print("Cannot perform Log-Rank Test: Missing 'ERL' or 'Programmatic' data.")
        return

    # Prepare data for Group 1 (ERL) and Group 2 (Programmatic)
    groups = {'ERL': results['ERL'], 'Programmatic': results['Programmatic']}
    
    # Extract times and events
    # Event: 1 if died (steps < MAX_STEPS), 0 if censored (steps == MAX_STEPS)
    data = {}
    for name, res in groups.items():
        steps = np.array([r[0] for r in res])
        events = (steps < MAX_STEPS).astype(int)
        data[name] = {'steps': steps, 'events': events}

    # Combine all unique event times from both groups
    all_steps = np.concatenate([data['ERL']['steps'], data['Programmatic']['steps']])
    all_events = np.concatenate([data['ERL']['events'], data['Programmatic']['events']])
    
    # Only consider times where at least one event (death) occurred
    event_times = np.unique(all_steps[all_events == 1])
    event_times.sort()
    
    if len(event_times) == 0:
        print("No death events occurred in either group. Cannot perform Log-Rank Test.")
        return

    # Calculate O_i (Observed) and E_i (Expected) for each group
    # We only need to calculate for one group (say ERL), the other is complementary
    
    O_1_total = 0 # Total observed deaths in Group 1
    E_1_total = 0 # Total expected deaths in Group 1
    V_total = 0   # Total variance
    
    for t in event_times:
        # Group 1 (ERL)
        n_1j = np.sum(data['ERL']['steps'] >= t) # Number at risk
        d_1j = np.sum((data['ERL']['steps'] == t) & (data['ERL']['events'] == 1)) # Deaths
        
        # Group 2 (Programmatic)
        n_2j = np.sum(data['Programmatic']['steps'] >= t)
        d_2j = np.sum((data['Programmatic']['steps'] == t) & (data['Programmatic']['events'] == 1))
        
        # Total
        n_j = n_1j + n_2j3
        d_j = d_1j + d_2j
        
        if n_j == 0: continue
            
        # Expected deaths for Group 1
        E_1j = n_1j * d_j / n_j
        
        # Variance contribution
        # V_j = (n_1j * n_2j * d_j * (n_j - d_j)) / (n_j^2 * (n_j - 1))
        if n_j > 1:
            V_j = (n_1j * n_2j * d_j * (n_j - d_j)) / (n_j**2 * (n_j - 1))
        else:
            V_j = 0
            
        O_1_total += d_1j
        E_1_total += E_1j
        V_total += V_j

    # Test Statistic Z
    # Z = (O_1 - E_1) / sqrt(V)
    if V_total > 0:
        Z = (O_1_total - E_1_total) / np.sqrt(V_total)
        Chi_sq = Z**2
        
        print(f"Observed Deaths (ERL): {O_1_total}")
        print(f"Expected Deaths (ERL): {E_1_total:.2f}")
        print(f"Chi-squared Statistic: {Chi_sq:.4f}")
        
        # Critical value for df=1, alpha=0.05 is 3.841
        critical_value = 3.841
        p_value_approx = " < 0.05" if Chi_sq > critical_value else " >= 0.05"
        significance = "Significant" if Chi_sq > critical_value else "Not Significant"
        
        print(f"Critical Value (alpha=0.05): {critical_value}")
        print(f"Result: {significance} difference (p{p_value_approx})")
        print("Null Hypothesis: Survival curves are identical.")
    else:
        print("Variance is zero. Cannot calculate Z-statistic.")


def plot_kaplan_meier(results):
    plt.figure(figsize=(10, 6))
    
    for strategy, data in results.items():
        # Extract steps and determine if event occurred (death) or censored (survived max steps)
        steps = np.array([d[0] for d in data])
        events = (steps < MAX_STEPS).astype(int) # 1 if died, 0 if survived (censored)
        
        # Sort by time
        sorted_indices = np.argsort(steps)
        sorted_steps = steps[sorted_indices]
        sorted_events = events[sorted_indices]
        
        # Unique times where events occurred
        unique_times = np.unique(sorted_steps)
        
        # Calculate S(t)
        survival_probs = [1.0]
        times = [0]
        
        current_survival = 1.0
        n_total = len(steps)
        
        for t in unique_times:
            # n_i: number at risk (survived >= t)
            n_i = np.sum(sorted_steps >= t)
            
            # d_i: number of deaths at time t
            d_i = np.sum((sorted_steps == t) & (sorted_events == 1))
            
            if n_i > 0:
                current_survival *= (1 - d_i / n_i)
            
            times.append(t)
            survival_probs.append(current_survival)
            
        # Step plot
        label = STRATEGY_LABELS.get(strategy, strategy)
        plt.step(times, survival_probs, where='post', label=label)
        
    plt.title(f"Kaplan–Meier Survival Curves Across Strategies (N={TRIALS_PER_STRATEGY} per strategy)")
    plt.xlabel('Time (Steps)')
    plt.ylabel('Survival Probability S(t)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = 'kaplan_meier_survival_curve.png'
    plt.savefig(output_file)
    print(f"Kaplan-Meier plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    run_experiments()
