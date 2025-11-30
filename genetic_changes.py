import ERL
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MAX_STEPS = 20000
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
    print("Starting simulation to reproduce Figure 12...")
    ERL.run_simulation(strategy='ERL', visualize=False, max_steps=MAX_STEPS, step_callback=data_collector)
    
    print("Simulation finished. Plotting results...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(history['steps'], history['plant_eval'], label='Plant-Evaluation', color='green', linestyle='-')
    plt.plot(history['steps'], history['plant_action'], label='Plant-Action', color='green', linestyle='--')
    plt.plot(history['steps'], history['carnivore_eval'], label='Carnivore-Evaluation', color='red', linestyle='-')
    plt.plot(history['steps'], history['carnivore_action'], label='Carnivore-Action', color='red', linestyle='--')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Average Weight Value')
    plt.title('Genetic Changes Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'genetic_changes.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
