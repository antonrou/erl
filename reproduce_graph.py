import ERL
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuration
STRATEGIES = ['ERL', 'E', 'L', 'F', 'B']
TRIALS_PER_STRATEGY = 10
MAX_STEPS = 20000 # Increased to capture longer lifetimes

def run_experiments():
    results = {s: [] for s in STRATEGIES}
    
    # Setup Real-time Plot
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
    
    print(f"Running {TRIALS_PER_STRATEGY} trials for each strategy...")
    
    for strategy in STRATEGIES:
        print(f"\n--- Strategy: {strategy} ---")
        for i in range(TRIALS_PER_STRATEGY):
            start_time = time.time()
            extinction_step = ERL.run_simulation(strategy=strategy, visualize=True, max_steps=MAX_STEPS)
            duration = time.time() - start_time
            results[strategy].append(extinction_step)
            print(f"Trial {i+1}/{TRIALS_PER_STRATEGY}: Extinction at step {extinction_step} ({duration:.2f}s)")
            
            # Update Plot
            lifetimes = np.sort(results[strategy])
            y = np.arange(1, len(lifetimes) + 1) / len(lifetimes) * 100
            lines[strategy].set_data(lifetimes, y)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
    plt.ioff()
    plt.savefig('population_lifetimes.png')
    print("Plot saved to population_lifetimes.png")
    plt.show()
    return results

if __name__ == "__main__":
    run_experiments()
