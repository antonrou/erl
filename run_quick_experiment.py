import ERL
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing

# Configuration for Quick Verification
STRATEGIES = ['ERL', 'E', 'L', 'F', 'B']
TRIALS_PER_STRATEGY = 10 # Increased for better distribution data
MAX_STEPS = 2000 # Reduced from 1,000,000

def run_single_trial(strategy, trial_num, seed_offset):
    # Use trial_num + offset as seed for reproducibility within a run, but variance across runs
    current_seed = trial_num + seed_offset
    # print(f"[{strategy}] Starting Trial {trial_num+1}/{TRIALS_PER_STRATEGY} (Seed: {current_seed})")
    steps = ERL.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS, seed=current_seed)
    return steps

def run_experiments():
    results = {s: [] for s in STRATEGIES}
    
    print(f"Running {TRIALS_PER_STRATEGY} trials (max {MAX_STEPS} steps) for each strategy using {multiprocessing.cpu_count()} CPU cores...")
    
    # Submit ALL tasks at once
    with multiprocessing.Pool() as pool:
        all_async_results = []
        
        # Generate a random seed offset for this run
        seed_offset = int(time.time()) % 10000
        print(f"Using seed offset: {seed_offset}")

        for strategy in STRATEGIES:
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
                        steps = res.get()
                        results[strategy].append(steps)
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
    for s in STRATEGIES:
        avg = np.mean(results[s]) if results[s] else 0
        med = np.median(results[s]) if results[s] else 0
        survived_full = sum(1 for x in results[s] if x >= MAX_STEPS)
        print(f"{s}: Avg={avg:.1f}, Median={med:.1f}, Survived Full Duration={survived_full}/{TRIALS_PER_STRATEGY}")
    
    plot_results(results)

def plot_results(results):
    num_strategies = len(results)
    fig, axes = plt.subplots(num_strategies, 1, figsize=(10, 4 * num_strategies), sharex=True, sharey=True)
    
    if num_strategies == 1:
        axes = [axes]
    
    for idx, (strategy, steps) in enumerate(results.items()):
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
