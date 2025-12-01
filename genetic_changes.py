import ERL
import matplotlib.pyplot as plt
import numpy as np
import run_quick_experiment
import random

# Configuration
MAX_STEPS = run_quick_experiment.MAX_STEPS
DATA_COLLECTION_INTERVAL = 100

# Indices for inputs
# N, E, S, W
PLANT_INDICES = [2, 8, 14, 20]
CARNIVORE_INDICES = [4, 10, 16, 22]

# Data storage
history = {
    'steps': [],
    'plant_eval': [],
    'plant_action': [],
    'carnivore_eval': [],
    'carnivore_action': []
}

def data_collector(step, world):
    if step % DATA_COLLECTION_INTERVAL != 0:
        return
        
    agents = world.agents
    if not agents:
        return

    # Collect weights from all living agents
    plant_eval_vals = []
    plant_action_vals = []
    carnivore_eval_vals = []
    carnivore_action_vals = []
    
    for agent in agents:
        # Eval weights (28x1)
        # We take the average weight for the specific input type across all directions
        # For Eval, it's just the weight at that index
        pe = np.mean([agent.w_eval[i][0] for i in PLANT_INDICES])
        ce = np.mean([agent.w_eval[i][0] for i in CARNIVORE_INDICES])
        plant_eval_vals.append(pe)
        carnivore_eval_vals.append(ce)
        
        # Action weights (28x2)
        # We want to know the magnitude of influence? 
        # Figure 12 says "Genetic changes". 
        # Usually this means the raw weight value.
        # Since there are 2 outputs (bits), we can average the absolute values or just the raw values.
        # If we average raw values, they might cancel out if one bit is + and other is -.
        # However, for "Action", maybe we care about the magnitude of reaction?
        # Let's try averaging the absolute values to see "sensitivity".
        # Or maybe just the first bit?
        # Let's stick to average of absolute weights for now to represent "strength of reaction".
        # Actually, let's look at the paper description again if possible.
        # "Values over time of fitness models". Fitness models usually refers to the Evaluation Network.
        # But the figure has "Plant-Action" and "Carnivore-Action".
        # Let's assume average absolute weight for Action to show "interest".
        
        pa = np.mean([np.mean(np.abs(agent.w_action[i])) for i in PLANT_INDICES])
        ca = np.mean([np.mean(np.abs(agent.w_action[i])) for i in CARNIVORE_INDICES])
        plant_action_vals.append(pa)
        carnivore_action_vals.append(ca)
        
    # Store population averages
    history['steps'].append(step)
    history['plant_eval'].append(np.mean(plant_eval_vals))
    history['plant_action'].append(np.mean(plant_action_vals))
    history['carnivore_eval'].append(np.mean(carnivore_eval_vals))
    history['carnivore_action'].append(np.mean(carnivore_action_vals))

def main():
    print("Running experiments to find a successful ERL trial...")
    # Run only ERL strategy to save time
    results = run_quick_experiment.run_experiments(strategies=['ERL'])
    
    erl_results = results['ERL']
    # Filter for trials that survived until MAX_STEPS
    survivors = [seed for (steps, seed) in erl_results if steps >= MAX_STEPS]
    
    if not survivors:
        print(f"No ERL trials survived until {MAX_STEPS} steps. Cannot reproduce Figure 12 with a successful agent.")
        # Optional: Pick the longest lasting one?
        # For now, let's just pick the best one we have.
        best_trial = max(erl_results, key=lambda x: x[0])
        print(f"Falling back to the best trial (Steps: {best_trial[0]}, Seed: {best_trial[1]})")
        selected_seed = best_trial[1]
    else:
        selected_seed = random.choice(survivors)
        print(f"Found {len(survivors)} successful trials. Selected seed: {selected_seed}")

    print(f"Starting simulation with seed {selected_seed} to reproduce Figure 12...")
    ERL.run_simulation(strategy='ERL', visualize=False, max_steps=MAX_STEPS, seed=selected_seed, step_callback=data_collector)
    
    print("Simulation finished. Plotting results...")
    
    # Plot Plants
    plt.figure(figsize=(10, 6))
    plt.plot(history['steps'], history['plant_eval'], label='Plant-Evaluation', color='green', linestyle='-')
    plt.plot(history['steps'], history['plant_action'], label='Plant-Action', color='green', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Weight Value')
    plt.title('Genetic Changes Over Time (Plants)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('genetic_changes_plants.png')
    print("Plants plot saved to genetic_changes_plants.png")

    # Plot Carnivores
    plt.figure(figsize=(10, 6))
    plt.plot(history['steps'], history['carnivore_eval'], label='Carnivore-Evaluation', color='red', linestyle='-')
    plt.plot(history['steps'], history['carnivore_action'], label='Carnivore-Action', color='red', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Weight Value')
    plt.title('Genetic Changes Over Time (Carnivores)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('genetic_changes_carnivores.png')
    print("Carnivores plot saved to genetic_changes_carnivores.png")

if __name__ == "__main__":
    main()
