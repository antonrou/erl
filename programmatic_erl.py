import random
import numpy as np
import math
from ERL import Entity, TYPE_AGENT, INPUT_SIZE, DIRS, MAX_ENERGY, MAX_HEALTH, ENERGY_COST_MOVE, ENERGY_GAIN_PLANT, ENERGY_GAIN_MEAT, DAMAGE_CARNIVORE, DAMAGE_WALL, REPRODUCE_ENERGY_THRESHOLD, REPRODUCE_COST, WORLD_WIDTH, WORLD_HEIGHT, Plant, Tree, Wall, Carnivore, TYPE_EMPTY, TYPE_WALL, TYPE_PLANT, TYPE_TREE, TYPE_CARNIVORE, AGENT_VIEW_DIST, get_distance_value, ERLAgent, World, LEARNING_RATE_POS, Visualizer

# Constants for Programmatic ERL
K_CLAUSES = 3
ACTION_SPACE_SIZE = 4 # N, E, S, W
INPUT_DIM = INPUT_SIZE

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x)) # Stability
    return e_x / e_x.sum()

class ProgrammaticGenome:
    def __init__(self):
        # Action Network Parameters
        # List of K clauses. Each clause:
        # w: vector of size INPUT_DIM
        # tau: scalar
        # logits: vector of size ACTION_SPACE_SIZE
        self.action_clauses = []
        for _ in range(K_CLAUSES):
            self.action_clauses.append({
                'w': np.random.uniform(-0.1, 0.1, INPUT_DIM),
                'tau': np.random.uniform(-0.1, 0.1),
                'logits': np.random.uniform(-0.1, 0.1, ACTION_SPACE_SIZE)
            })
            
        # Evaluation Network Parameters
        # List of K clauses. Each clause:
        # w: vector of size INPUT_DIM
        # tau: scalar
        # u: scalar (goodness)
        self.eval_clauses = []
        for _ in range(K_CLAUSES):
            self.eval_clauses.append({
                'w': np.random.uniform(-0.1, 0.1, INPUT_DIM),
                'tau': np.random.uniform(-0.1, 0.1),
                'u': np.random.uniform(0, 1) # Goodness between 0 and 1? Prompt says Et in (0,1)
            })
        self.eval_bias = np.random.uniform(-0.1, 0.1)

    def mutate(self, rate=0.2, perturbation_std=0.2):
        # Mutate Action Parameters
        for clause in self.action_clauses:
            if random.random() < rate:
                clause['w'] += np.random.normal(0, perturbation_std, INPUT_DIM)
            if random.random() < rate:
                clause['tau'] += np.random.normal(0, perturbation_std)
            if random.random() < rate:
                clause['logits'] += np.random.normal(0, perturbation_std, ACTION_SPACE_SIZE)
                
        # Mutate Eval Parameters
        for clause in self.eval_clauses:
            if random.random() < rate:
                clause['w'] += np.random.normal(0, perturbation_std, INPUT_DIM)
            if random.random() < rate:
                clause['tau'] += np.random.normal(0, perturbation_std)
            if random.random() < rate:
                clause['u'] += np.random.normal(0, perturbation_std)
        
        if random.random() < rate:
            self.eval_bias += np.random.normal(0, perturbation_std)

    def crossover(self, other):
        # Clause-level crossover
        child = ProgrammaticGenome()
        
        # Action clauses
        for i in range(K_CLAUSES):
            if random.random() < 0.5:
                child.action_clauses[i] = copy_clause(self.action_clauses[i])
            else:
                child.action_clauses[i] = copy_clause(other.action_clauses[i])
                
        # Eval clauses
        for i in range(K_CLAUSES):
            if random.random() < 0.5:
                child.eval_clauses[i] = copy_clause(self.eval_clauses[i])
            else:
                child.eval_clauses[i] = copy_clause(other.eval_clauses[i])
                
        child.eval_bias = self.eval_bias if random.random() < 0.5 else other.eval_bias
        return child

def copy_clause(clause):
    new_clause = {}
    for k, v in clause.items():
        if isinstance(v, np.ndarray):
            new_clause[k] = v.copy()
        else:
            new_clause[k] = v
    return new_clause

class ProgrammaticERLAgent(ERLAgent):
    def __init__(self, x, y, genome=None):
        # Skip ERLAgent.__init__ to avoid bitstring genome decoding
        Entity.__init__(self, x, y, TYPE_AGENT)
        self.genome = genome if genome else ProgrammaticGenome()
        
        self.energy = MAX_ENERGY * 0.5
        self.health = MAX_HEALTH
        self.in_tree = False
        self.age = 0
        
        # State for learning
        self.prev_eval = 0.0
        self.just_born = True
        
        # Learning Hyperparameters
        self.alpha = 0.05 # Learning rate
        self.baseline = 0.0 # Reinforcement baseline (moving average?)
        self.baseline_alpha = 0.1 # For updating baseline
        
    def get_inputs(self, world):
        # Copied/Adapted from ERL.py ERLAgent.get_inputs
        # We need to ensure we use the same input structure
        inputs = np.zeros(INPUT_DIM)
        grid = world.grid
        
        for i, (dx, dy) in enumerate(DIRS): # N, E, S, W
            closest_type = TYPE_EMPTY
            closest_dist = AGENT_VIEW_DIST + 1
            
            tx, ty = self.x, self.y
            for d in range(1, AGENT_VIEW_DIST + 1):
                tx += dx
                ty += dy
                
                if not (0 <= tx < WORLD_WIDTH and 0 <= ty < WORLD_HEIGHT):
                    closest_type = TYPE_WALL
                    closest_dist = d
                    break
                
                occ = grid[tx][ty]
                if occ:
                    closest_type = occ.type_id
                    closest_dist = d
                    break
            
            val = get_distance_value(closest_dist, AGENT_VIEW_DIST)
            
            base_idx = i * 6
            type_idx = base_idx + closest_type
            inputs[type_idx] = val
            
        offset = 24
        inputs[offset] = self.health
        inputs[offset+1] = self.energy / MAX_ENERGY
        inputs[offset+2] = 1.0 if self.in_tree else 0.0
        inputs[offset+3] = 1.0 # Bias
        
        return inputs

    def compute_action_policy(self, state):
        # Returns:
        # - policy_probs: array of shape (ACTION_SPACE_SIZE,)
        # - clause_probs: array of shape (K_CLAUSES,) - p(ci|s)
        # - action_probs_per_clause: array of shape (K_CLAUSES, ACTION_SPACE_SIZE) - p(a|ci, s)
        # - gate_activations: array of shape (K_CLAUSES,) - gi(s)
        
        gate_activations = np.zeros(K_CLAUSES)
        clause_probs = np.zeros(K_CLAUSES)
        action_probs_per_clause = np.zeros((K_CLAUSES, ACTION_SPACE_SIZE))
        
        remaining_prob = 1.0
        
        for i in range(K_CLAUSES):
            clause = self.genome.action_clauses[i]
            w = clause['w']
            tau = clause['tau']
            logits = clause['logits']
            
            # Gate activation
            g_val = sigmoid(np.dot(w, state) - tau)
            gate_activations[i] = g_val
            
            # Clause firing probability
            # p(ci|s) = gi(s) * Product_{j<i} (1 - gj(s))
            # This is equivalent to remaining_prob * g_val
            p_ci = remaining_prob * g_val
            clause_probs[i] = p_ci
            
            remaining_prob *= (1.0 - g_val)
            
            # Action distribution within clause
            p_a_given_ci = softmax(logits)
            action_probs_per_clause[i] = p_a_given_ci
            
        # Complete policy
        # pi(a|s) = Sum_i p(ci|s) * p(a|ci, s)
        policy_probs = np.zeros(ACTION_SPACE_SIZE)
        for i in range(K_CLAUSES):
            policy_probs += clause_probs[i] * action_probs_per_clause[i]
            
        # Normalize policy_probs just in case (though math says it should sum to <= 1, 
        # actually if we don't have a "default" clause, it might not sum to 1 if remaining_prob > 0.
        # The prompt implies p(ci|s) definition covers it? 
        # "This indicates that the all clauses before i are not fired until the clause i itself."
        # If all gates are 0, sum is 0. 
        # Usually soft decision lists have a final default clause or ensure sum is 1.
        # The prompt doesn't mention a default clause.
        # However, if we strictly follow the formula, we might get sum < 1.
        # Let's normalize if sum > 0, else uniform.
        
        total_p = np.sum(policy_probs)
        if total_p > 1e-9:
            policy_probs /= total_p
        else:
            policy_probs = np.ones(ACTION_SPACE_SIZE) / ACTION_SPACE_SIZE
            
        return policy_probs, clause_probs, action_probs_per_clause, gate_activations

    def compute_evaluation(self, state):
        # Returns Et
        # Also returns intermediate values if needed for something (not needed for Eval update since it's fixed)
        
        val_sum = 0.0
        remaining_prob = 1.0
        
        for i in range(K_CLAUSES):
            clause = self.genome.eval_clauses[i]
            w = clause['w']
            tau = clause['tau']
            u = clause['u']
            
            g_val = sigmoid(np.dot(w, state) - tau)
            p_ci = remaining_prob * g_val
            remaining_prob *= (1.0 - g_val)
            
            val_sum += p_ci * u
            
        Et = sigmoid(val_sum + self.genome.eval_bias)
        return Et

    def step(self, world):
        self.age += 1
        current_input = self.get_inputs(world)
        
        # 1. Compute Evaluation
        current_eval = self.compute_evaluation(current_input)
        
        # 2. Learning (REINFORCE)
        if not self.just_born:
            # Reward
            r = current_eval - self.prev_eval
            
            # Update Baseline (Simple moving average)
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * r
            advantage = r - self.baseline
            
            # We need to update Action Parameters based on the PREVIOUS action and state?
            # REINFORCE typically uses the gradient of the log-prob of the action taken.
            # So we need to store the gradients or the state/action from the previous step.
            # The prompt says: "At each timestep t, the environment parameter change Delta theta ... is ... alpha * (Rt - bt) * grad log(pi(a|s))"
            # Usually REINFORCE updates based on the action taken at time t that resulted in reward.
            # Here Rt = Et - E_{t-1}. This reward is for the transition from t-1 to t.
            # So we should update the policy that generated the action at t-1.
            # So we need self.prev_input and self.prev_action_idx.
            
            if hasattr(self, 'prev_input') and hasattr(self, 'prev_action_idx'):
                self.learn(self.prev_input, self.prev_action_idx, advantage)
        
        # 3. Action Selection
        policy_probs, _, _, _ = self.compute_action_policy(current_input)
        
        # Sample action
        action_idx = np.random.choice(ACTION_SPACE_SIZE, p=policy_probs)
        
        # Store state for next step
        self.prev_input = current_input
        self.prev_action_idx = action_idx
        self.prev_eval = current_eval
        self.just_born = False
        
        # 4. Execute Action
        self.perform_action(world, action_idx)
        
    def learn(self, state, action_idx, advantage):
        # Compute gradients for Action Network at 'state' for 'action_idx'
        policy_probs, clause_probs, action_probs_per_clause, gate_activations = self.compute_action_policy(state)
        
        # pi(a|s)
        pi_a = policy_probs[action_idx]
        if pi_a < 1e-10: pi_a = 1e-10 # Avoid division by zero
        
        # Posterior rho_k(a) = p(ck | a, s) = p(ck|s) * p(a|ck, s) / pi(a|s)
        rho = (clause_probs * action_probs_per_clause[:, action_idx]) / pi_a
        
        # 1. Update Logits Li
        # Grad_Li log pi = rho_i(a) * (1_a - Si)
        # Si is action_probs_per_clause[i]
        for i in range(K_CLAUSES):
            Si = action_probs_per_clause[i]
            grad_Li = - rho[i] * Si # Term for all j != a
            grad_Li[action_idx] += rho[i] # Add rho[i] * 1 for j == a
            
            # Update
            self.genome.action_clauses[i]['logits'] += self.alpha * advantage * grad_Li
            
        # 2. Update Gates wi, tau_i
        # Grad_wi log pi = [ rho_i(a) * (1 - Gi) - Gi * Sum_{k>i} rho_k(a) ] * s
        # Grad_tau_i log pi = - [ ... ] (same bracket)
        
        # Calculate Sum_{k>i} rho_k(a) efficiently
        # suffix_sum_rho[i] = Sum_{k=i+1}^{K-1} rho[k]
        suffix_sum_rho = np.zeros(K_CLAUSES)
        current_sum = 0.0
        for k in range(K_CLAUSES - 1, -1, -1):
            suffix_sum_rho[k] = current_sum
            current_sum += rho[k]
            
        for i in range(K_CLAUSES):
            Gi = gate_activations[i]
            term = rho[i] * (1.0 - Gi) - Gi * suffix_sum_rho[i]
            
            grad_w = term * state
            grad_tau = -term
            
            self.genome.action_clauses[i]['w'] += self.alpha * advantage * grad_w
            self.genome.action_clauses[i]['tau'] += self.alpha * advantage * grad_tau

    def perform_action(self, world, action_idx):
        # Copied from ERLAgent.perform_action
        dx, dy = DIRS[action_idx]
        nx, ny = self.x + dx, self.y + dy
        
        if 0 <= nx < WORLD_WIDTH and 0 <= ny < WORLD_HEIGHT:
            occ = world.get_occupant(nx, ny)
            
            if occ is None:
                if self.in_tree: self.in_tree = False
                world.move_entity(self, nx, ny)
                self.energy -= ENERGY_COST_MOVE
                
            elif isinstance(occ, Plant):
                world.remove_entity(occ)
                self.energy += ENERGY_GAIN_PLANT
                if self.energy > MAX_ENERGY: self.energy = MAX_ENERGY
                world.move_entity(self, nx, ny)
                
            elif isinstance(occ, Tree):
                if occ.occupant is None:
                    world.move_entity(self, nx, ny)
                    self.in_tree = True
                    occ.occupant = self
                else:
                    self.energy -= ENERGY_COST_MOVE
                    
            elif isinstance(occ, Wall):
                self.health -= DAMAGE_WALL
                self.energy -= ENERGY_COST_MOVE
                
            elif isinstance(occ, Carnivore):
                if occ.dead:
                    world.remove_entity(occ)
                    self.energy += ENERGY_GAIN_MEAT
                else:
                    occ.health -= DAMAGE_CARNIVORE
                    occ.last_damage_source = 'agent'
                    self.energy -= ENERGY_COST_MOVE
                
            elif isinstance(occ, Entity) and occ.type_id == TYPE_AGENT: # Check type_id for generic agent
                if occ.dead:
                    world.remove_entity(occ)
                    self.energy += ENERGY_GAIN_MEAT
                else:
                    occ.health -= DAMAGE_CARNIVORE
                    occ.last_damage_source = 'agent'
                    self.energy -= ENERGY_COST_MOVE
        else:
            self.health -= DAMAGE_WALL
            self.energy -= ENERGY_COST_MOVE
            
        self.check_reproduction(world)

    def check_reproduction(self, world):
        if self.energy > REPRODUCE_ENERGY_THRESHOLD:
            child_genome = None
            
            # Assuming ERL strategy (Evolution + Learning)
            mate = world.find_closest_mate(self)
            if mate and isinstance(mate, ProgrammaticERLAgent):
                child_genome = self.genome.crossover(mate.genome)
            else:
                # Clone
                child_genome = ProgrammaticGenome()
                # Copy current genome
                child_genome.action_clauses = [copy_clause(c) for c in self.genome.action_clauses]
                child_genome.eval_clauses = [copy_clause(c) for c in self.genome.eval_clauses]
                child_genome.eval_bias = self.genome.eval_bias
                
            child_genome.mutate()
            
            self.energy -= REPRODUCE_COST
            world.spawn_agent_near(self.x, self.y, child_genome)
            
        if self.energy <= 0 or self.health <= 0:
            self.dead = True

class ProgrammaticWorld(World):
    def __init__(self):
        super().__init__()
        
    def init_world(self):
        self.grid = [[None for _ in range(WORLD_HEIGHT)] for _ in range(WORLD_WIDTH)]
        self.plants = []
        self.trees = []
        self.walls = []
        self.carnivores = []
        self.agents = []
        self.step_count = 0
        
        # Walls around edges
        for x in range(WORLD_WIDTH):
            self.add_entity(Wall(x, 0))
            self.add_entity(Wall(x, WORLD_HEIGHT-1))
        for y in range(1, WORLD_HEIGHT-1):
            self.add_entity(Wall(0, y))
            self.add_entity(Wall(WORLD_WIDTH-1, y))
            
        # Random internal walls (scattered)
        for _ in range(50):
            self.add_entity(Wall(random.randint(1,98), random.randint(1,98)))
            
        # Trees
        for _ in range(30):
            self.add_entity(Tree(random.randint(1,98), random.randint(1,98)))
            
        # Plants
        for _ in range(200):
            self.add_entity(Plant(random.randint(1,98), random.randint(1,98)))
            
        # Initial Agents
        for _ in range(100):
            # Random spawn distribution
            self.spawn_agent_near(random.randint(1,98), random.randint(1,98), ProgrammaticGenome())
            
        # Initial Carnivores
        for _ in range(5):
            self.add_entity(Carnivore(random.randint(1,98), random.randint(1,98)))

    def spawn_agent_near(self, x, y, genome):
        # Try adjacent cells
        dirs = list(DIRS)
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.add_entity(ProgrammaticERLAgent(nx, ny, genome)):
                return
        # If crowded, random spot
        self.add_entity(ProgrammaticERLAgent(random.randint(1,98), random.randint(1,98), genome))


# Indices for inputs (N, E, S, W) - Same as ERL.py
PLANT_INDICES = [2, 8, 14, 20]
CARNIVORE_INDICES = [4, 10, 16, 22]

def get_population_metrics(world):
    agents = world.agents
    if not agents:
        return None
        
    plant_eval_vals = []
    plant_action_vals = []
    carnivore_eval_vals = []
    carnivore_action_vals = []
    
    for agent in agents:
        # Average weight magnitude for these indices across all clauses
        
        # Eval
        pe_sum = 0
        ce_sum = 0
        for clause in agent.genome.eval_clauses:
            w = clause['w']
            pe_sum += np.mean([abs(w[i]) for i in PLANT_INDICES])
            ce_sum += np.mean([abs(w[i]) for i in CARNIVORE_INDICES])
        plant_eval_vals.append(pe_sum / len(agent.genome.eval_clauses))
        carnivore_eval_vals.append(ce_sum / len(agent.genome.eval_clauses))

        # Action
        pa_sum = 0
        ca_sum = 0
        for clause in agent.genome.action_clauses:
            w = clause['w']
            pa_sum += np.mean([abs(w[i]) for i in PLANT_INDICES])
            ca_sum += np.mean([abs(w[i]) for i in CARNIVORE_INDICES])
        plant_action_vals.append(pa_sum / len(agent.genome.action_clauses))
        carnivore_action_vals.append(ca_sum / len(agent.genome.action_clauses))
            
    return {
        'plant_eval': np.mean(plant_eval_vals),
        'plant_action': np.mean(plant_action_vals),
        'carnivore_eval': np.mean(carnivore_eval_vals),
        'carnivore_action': np.mean(carnivore_action_vals)
    }

def run_simulation(strategy='Programmatic', visualize=True, max_steps=10000, seed=None, step_callback=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    world = ProgrammaticWorld()
    vis = None
    if visualize:
        vis = Visualizer(WORLD_WIDTH, WORLD_HEIGHT)
    
    # History storage
    history = {
        'steps': [],
        'plant_eval': [],
        'plant_action': [],
        'carnivore_eval': [],
        'carnivore_action': []
    }
    
    # Run loop
    try:
        for t in range(1, max_steps + 1):
            world.update()
            
            if step_callback:
                step_callback(t, world)
            
            # Collect metrics every 100 steps
            if t % 100 == 0:
                metrics = get_population_metrics(world)
                if metrics:
                    history['steps'].append(t)
                    history['plant_eval'].append(metrics['plant_eval'])
                    history['plant_action'].append(metrics['plant_action'])
                    history['carnivore_eval'].append(metrics['carnivore_eval'])
                    history['carnivore_action'].append(metrics['carnivore_action'])
            
            if visualize and t % 10 == 0:
                vis.update(world)
            
            if t % 10000 == 0:
                if not visualize:
                    print(f"[{strategy}] Step {t}...", flush=True)

            if t % 100 == 0:
                if visualize:
                    avg_fitness = 0
                    if world.agents:
                        avg_fitness = sum(a.energy for a in world.agents) / len(world.agents)
                    print(f"Step {t}: Agents={len(world.agents)} Carnivores={len(world.carnivores)} Plants={len(world.plants)} AvgEnergy={avg_fitness:.1f}")
                
            if len(world.agents) == 0:
                if visualize: print("Extinction event.")
                return t, history
        
        return max_steps, history
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        return t, history
    finally:
        if vis:
            vis.close()

if __name__ == "__main__":
    run_simulation()
