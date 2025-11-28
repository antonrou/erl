import ERL
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuration
STRATEGIES = ['ERL', 'E', 'L', 'F', 'B']
TRIALS_PER_STRATEGY = 20
MAX_STEPS = 20000 # Increased to capture longer lifetimes

import multiprocessing
import functools

def run_single_trial(strategy, trial_idx):
    # Wrapper to run a single trial
    # visualize=False is mandatory for parallel execution to avoid GUI conflicts
    print(f"[{strategy}] Starting Trial {trial_idx+1}/{TRIALS_PER_STRATEGY}")
    steps = ERL.run_simulation(strategy=strategy, visualize=False, max_steps=MAX_STEPS)
    return steps

def run_experiments():
    results = {s: [] for s in STRATEGIES}
    
    # Setup Real-time Plot (Runs in main process)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')
    ax.set_xlabel('Extinction Time (log scale)')
    ax.set_ylabel('% of Initial Populations')
    ax.set_title('Distribution of Population Lifetimes')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlim(1, MAX_STEPS * 1.5)
    ax.set_ylim(0, 105)
    
    styles = {
        'ERL': '-',   # Solid
        'E': ':',     # Dotted
        'L': '--',    # Dashed
        'F': '-.',    # Dash-dot
        'B': (0, (3, 5, 1, 5)) # Loose dash dot
    }
    
    lines = {}
    for s in STRATEGIES:
        line, = ax.plot([], [], linestyle=styles.get(s, '-'), label=s, linewidth=2)
        lines[s] = line
    
    ax.legend()
    plt.show(block=False)
    
    print(f"Running {TRIALS_PER_STRATEGY} trials for each strategy using {multiprocessing.cpu_count()} CPU cores...")
    
    # Submit ALL tasks at once to avoid blocking
    with multiprocessing.Pool() as pool:
        all_async_results = []
        
        # Launch all trials for all strategies
        for strategy in STRATEGIES:
            for i in range(TRIALS_PER_STRATEGY):
                res = pool.apply_async(run_single_trial, (strategy, i))
                all_async_results.append((strategy, i, res))
        
        start_time = time.time()
        total_trials = len(all_async_results)
        finished_count = 0
        
        while finished_count < total_trials:
            updated = False
            for idx, (strategy, trial_num, res) in enumerate(all_async_results):
                if res is None: continue
                
                if res.ready():
                    steps = res.get()
                    results[strategy].append(steps)
                    all_async_results[idx] = (strategy, trial_num, None) # Mark collected
                    finished_count += 1
                    print(f"[{strategy}] Trial {trial_num+1}/{TRIALS_PER_STRATEGY} finished: {steps} steps")
                    updated = True
            
            if updated:
                # Update Plot for all strategies
                for strategy in STRATEGIES:
                    if results[strategy]:
                        lifetimes = np.sort(results[strategy])
                        # Normalize by TOTAL trials per strategy
                        y = np.arange(1, len(lifetimes) + 1) / TRIALS_PER_STRATEGY * 100
                        lines[strategy].set_data(lifetimes, y)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            plt.pause(0.1) # Keep GUI alive
            
        duration = time.time() - start_time
        print(f"All experiments finished in {duration:.2f}s")
    
    plt.ioff()
    plt.savefig('population_lifetimes.png')
    print("Plot saved to population_lifetimes.png")
    plt.show()
    return results

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    run_experiments()
