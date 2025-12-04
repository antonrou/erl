import ERL
import programmatic_erl
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from scipy.stats import norm

# Configuration for Quick Verification
STRATEGIES = ['ERL', 'E', 'L', 'F', 'Programmatic', 'PE', 'PL', 'B']
TRIALS_PER_STRATEGY = 500 # Reduced for quick baseline
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
    perform_rmst_analysis(results)




def perform_rmst_analysis(results):
    """
    Perform Restricted Mean Survival Time (RMST) analysis.
    Calculates RMST, standard error, and compares Programmatic vs ERL.
    """
    print("\n--- Restricted Mean Survival Time (RMST) Analysis ---")
    if 'ERL' not in results or 'Programmatic' not in results:
        print("Note: 'ERL' or 'Programmatic' missing, skipping specific comparison.")

    tau = MAX_STEPS
    print(f"Time Horizon (tau): {tau} steps")
    
    analysis_data = {}
    
    # Calculate RMST for ALL strategies
    print(f"{'Strategy':<15} {'RMST':<10} {'SE':<10} {'95% CI':<20}")
    print("-" * 60)
    
    for strategy in results.keys():
        steps = np.array([r[0] for r in results[strategy]])
        if len(steps) == 0:
            continue
            
        events = (steps < MAX_STEPS).astype(int)
        
        kmf = KaplanMeierFitter()
        kmf.fit(steps, event_observed=events, label=strategy)
        
        # Calculate RMST
        rmst = restricted_mean_survival_time(kmf, t=tau)
        
        # Calculate Variance of RMST (Greenwood-type approximation)
        event_times = kmf.event_table.index[kmf.event_table['observed'] > 0]
        event_table = kmf.event_table.loc[event_times]
        
        var_rmst = 0
        survival_function = kmf.survival_function_
        
        for t_i in event_times:
            if t_i >= tau:
                break
                
            d_i = event_table.loc[t_i, 'observed']
            n_i = event_table.loc[t_i, 'at_risk']
            
            if n_i <= d_i:
                continue
                
            # Calculate area under S(t) from t_i to tau
            mask = (survival_function.index >= t_i) & (survival_function.index <= tau)
            times_in_range = survival_function.index[mask]
            
            if len(times_in_range) == 0:
                continue
            
            # Efficient calculation of area under step function
            relevant_times = survival_function.index.values
            relevant_times = relevant_times[relevant_times >= t_i]
            relevant_times = relevant_times[relevant_times < tau]
            relevant_times = np.append(relevant_times, tau)
            relevant_times = np.unique(relevant_times)
            
            area = 0
            for k in range(len(relevant_times) - 1):
                t_start = relevant_times[k]
                t_end = relevant_times[k+1]
                s_val = kmf.predict(t_start)
                area += s_val * (t_end - t_start)
            
            term = (area**2) * d_i / (n_i * (n_i - d_i))
            var_rmst += term
            
        se_rmst = np.sqrt(var_rmst)
        analysis_data[strategy] = {'rmst': rmst, 'se': se_rmst, 'n': len(steps)}
        
        ci_lower = rmst - 1.96 * se_rmst
        ci_upper = rmst + 1.96 * se_rmst
        print(f"{strategy:<15} {rmst:<10.2f} {se_rmst:<10.2f} ({ci_lower:.1f}, {ci_upper:.1f})")

    # Compare Programmatic vs ERL if both exist
    if 'Programmatic' in analysis_data and 'ERL' in analysis_data:
        rmst1 = analysis_data['Programmatic']['rmst']
        se1 = analysis_data['Programmatic']['se']
        
        rmst2 = analysis_data['ERL']['rmst']
        se2 = analysis_data['ERL']['se']
        
        diff = rmst1 - rmst2
        se_diff = np.sqrt(se1**2 + se2**2)
        
        z_score = diff / se_diff
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        print("-" * 60)
        print(f"Difference (Programmatic - ERL): {diff:.2f}")
        print(f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
        print(f"Z-score: {z_score:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: Statistically Significant Difference")
        else:
            print("Result: No Statistically Significant Difference")
    print("-" * 60)
    print("-" * 60)


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
        n_j = n_1j + n_2j
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
        
        kmf = KaplanMeierFitter()
        label = STRATEGY_LABELS.get(strategy, strategy)
        kmf.fit(steps, event_observed=events, label=label)
        
        # Plot with confidence bands (default in lifelines)
        kmf.plot_survival_function()
        
    plt.title(f"Kaplan–Meier Survival Curves (N={TRIALS_PER_STRATEGY} per strategy)\nwith 95% Confidence Bands (Greenwood)")
    plt.xlabel('Time (Steps)')
    plt.ylabel('Survival Probability S(t)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
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
