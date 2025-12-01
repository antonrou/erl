import ERL
import programmatic_erl
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing

# Configuration for Quick Verification
STRATEGIES = ['ERL', 'E', 'L', 'F', 'B', 'Programmatic']
TRIALS_PER_STRATEGY = 100 # Reduced for quick baseline
MAX_STEPS = 2000

# Shielding Parameters
SHIELDING_DELTA = 0.07
SHIELDING_TAU = 0.2 # 20% of time

def run_single_trial(strategy, trial_num, seed_offset):
    # Use trial_num + offset as seed for reproducibility within a run, but variance across runs
    current_seed = trial_num + seed_offset
    # print(f"[{strategy}] Starting Trial {trial_num+1}/{TRIALS_PER_STRATEGY} (Seed: {current_seed})")
    if strategy == 'Programmatic':
        steps, history = programmatic_erl.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS, seed=current_seed)
    else:
        steps, history = ERL.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS, seed=current_seed)
    return steps, current_seed, history

def calculate_shielding(history, delta=SHIELDING_DELTA, tau=SHIELDING_TAU):
    # Check if shielding occurs: Fact - Feval > delta for >= tau fraction of time
    # We check for Plants (primary task)
    
    # history lists are sampled every 100 steps.
    # We iterate through the collected points.
    
    steps = history['steps']
    if not steps:
        return False
        
    T = len(steps)
    shielding_count = 0
    
    for i in range(T):
        fact = history['carnivore_action'][i]
        feval = history['carnivore_eval'][i]
        
        diff = fact - feval
        if diff > delta:
            shielding_count += 1
            
    fraction = shielding_count / T
    return fraction >= tau

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
    
    # Calculate Shielding Prominence for ERL and Programmatic
    phi_values = {}
    for strategy_name in ['ERL', 'Programmatic']:
        if strategy_name in results:
            data = results[strategy_name]
            survivors = [r for r in data if r[0] >= MAX_STEPS]
            
            print(f"\n--- Shielding Prominence ({strategy_name}) ---")
            if survivors:
                shielding_runs = 0
                for steps, seed, history in survivors:
                    if calculate_shielding(history):
                        shielding_runs += 1
                
                phi = (shielding_runs / len(survivors)) * 100.0
                phi_values[strategy_name] = phi
                print(f"Total Survivors: {len(survivors)}")
                print(f"Shielding Occurred in: {shielding_runs}")
                print(f"Shielding Prominence (Phi): {phi:.1f}%")
            else:
                print(f"No {strategy_name} trials survived full duration. Cannot calculate shielding.")

    # Calculate Quality of Imitation (J)
    if 'ERL' in phi_values and 'Programmatic' in phi_values:
        phi_ann = phi_values['ERL']
        phi_prog = phi_values['Programmatic']
        J = - abs(phi_ann - phi_prog)
        print(f"\n--- Quality of Imitation ---")
        print(f"Phi_ann (ERL): {phi_ann:.1f}%")
        print(f"Phi_prog (Programmatic): {phi_prog:.1f}%")
        print(f"J = - | Phi_ann - Phi_prog | = {J:.1f}")

    plot_results(results)
    return results

def plot_results(results):
    num_strategies = len(results)
    fig, axes = plt.subplots(num_strategies, 1, figsize=(10, 4 * num_strategies), sharex=True, sharey=True)
    
    if num_strategies == 1:
        axes = [axes]
    
    for idx, (strategy, data) in enumerate(results.items()):
        steps = [d[0] for d in data]
        ax = axes[idx]
        
        # Create histogram
        # Bins: 20 bins spanning from 0 to MAX_STEPS
        bins = np.linspace(0, MAX_STEPS, 21)
        ax.hist(steps, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.set_title(f'Strategy: {strategy}')
        ax.set_ylabel('Frequency (Trials)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set x-ticks to match bins
        ax.set_xticks(bins)
        ax.tick_params(axis='x', rotation=45)
        
        # Add mean/median lines
        avg = np.mean(steps) if steps else 0
        med = np.median(steps) if steps else 0
        ax.axvline(avg, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg:.1f}')
        ax.axvline(med, color='green', linestyle='dashed', linewidth=1, label=f'Median: {med:.1f}')
        ax.legend()

    plt.xlabel('Steps Survived')
    plt.tight_layout()
    
    output_file = 'experiment_results_distribution.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    run_experiments()
