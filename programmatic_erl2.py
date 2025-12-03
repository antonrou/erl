import random
import numpy as np
import math
from ERL import Entity, TYPE_AGENT, INPUT_SIZE, DIRS, MAX_ENERGY, MAX_HEALTH, ENERGY_COST_MOVE, ENERGY_GAIN_PLANT, ENERGY_GAIN_MEAT, DAMAGE_CARNIVORE, DAMAGE_WALL, REPRODUCE_ENERGY_THRESHOLD, REPRODUCE_COST, WORLD_WIDTH, WORLD_HEIGHT, Plant, Tree, Wall, Carnivore, TYPE_EMPTY, TYPE_WALL, TYPE_PLANT, TYPE_TREE, TYPE_CARNIVORE, AGENT_VIEW_DIST, get_distance_value, ERLAgent, World, LEARNING_RATE_POS, Visualizer

# --- FAIRNESS CONFIGURATION ---
# K=2 gives the agent just enough hierarchy to be distinct from the Linear ANN,
# but simple enough to converge in 2,000 steps.
K_CLAUSES = 2
ACTION_SPACE_SIZE = 4 
INPUT_DIM = INPUT_SIZE

def sigmoid(x):
    # Clip x to prevent overflow/underflow warnings
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()

def copy_clause(clause):
    new_clause = {}
    for k, v in clause.items():
        if isinstance(v, np.ndarray):
            new_clause[k] = v.copy()
        else:
            new_clause[k] = v
    return new_clause

class ProgrammaticGenome:
    def __init__(self):
        # Action Network Parameters
        self.action_clauses = []
        for _ in range(K_CLAUSES):
            self.action_clauses.append({
                'w': np.random.uniform(-0.1, 0.1, INPUT_DIM),
                # FAIRNESS FIX 1: Open Gates Initialization
                # Negative tau ensures sigmoid(0 - tau) > 0.5.
                # Matches ANN's initial "transparent" behavior.
                'tau': np.random.uniform(-0.5, -0.1), 
                'logits': np.random.uniform(-0.1, 0.1, ACTION_SPACE_SIZE)
            })
            
        # Evaluation Network Parameters
        self.eval_clauses = []
        for _ in range(K_CLAUSES):
            self.eval_clauses.append({
                'w': np.random.uniform(-0.1, 0.1, INPUT_DIM),
                'tau': np.random.uniform(-0.5, -0.1), # Also open for evaluation
                'u': np.random.uniform(0, 1) 
            })
        self.eval_bias = np.random.uniform(-0.1, 0.1)

    def mutate(self, rate=0.05, perturbation_std=0.05):
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

class ProgrammaticERLAgent(ERLAgent):
    def __init__(self, x, y, genome=None):
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
        self.alpha = 0.05  # keep same to be comparable to CRBP
        # no baseline, no baseline_alpha
        
    def get_inputs(self, world):
        inputs = np.zeros(INPUT_DIM)
        grid = world.grid
        for i, (dx, dy) in enumerate(DIRS):
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
        inputs[offset+3] = 1.0 
        return inputs

    def compute_action_policy(self, state):
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
            
            # Clause probability
            p_ci = remaining_prob * g_val
            clause_probs[i] = p_ci
            
            remaining_prob *= (1.0 - g_val)
            
            # Action distribution
            p_a_given_ci = softmax(logits)
            action_probs_per_clause[i] = p_a_given_ci
            
        policy_probs = np.zeros(ACTION_SPACE_SIZE)
        for i in range(K_CLAUSES):
            policy_probs += clause_probs[i] * action_probs_per_clause[i]
            
        total_p = np.sum(policy_probs)
        if total_p > 1e-9:
            policy_probs /= total_p
        else:
            policy_probs = np.ones(ACTION_SPACE_SIZE) / ACTION_SPACE_SIZE
            
        return policy_probs, clause_probs, action_probs_per_clause, gate_activations

    def compute_evaluation(self, state):
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
        current_eval = self.compute_evaluation(current_input)

        # --- LEARNING STEP (CRBP-style sign(r) + early stopping) ---
        if not self.just_born and hasattr(self, 'prev_input') and hasattr(self, 'prev_action_idx'):
            # Same "physics reward": delta in internal evaluation
            r = current_eval - self.prev_eval

            # Only learn if something changed
            if abs(r) > 1e-6:
                # CRBP only cares about sign of r, not magnitude
                r_sign = 1.0 if r > 0 else -1.0

                # Mental rehearsal loop: up to 20 updates, but with early stopping
                for _ in range(20):
                    # One gradient step on the previous state/action
                    self.learn(self.prev_input, self.prev_action_idx, r_sign)

                    # Re-sample a temporary action from the updated policy on the same state
                    policy_probs, _, _, _ = self.compute_action_policy(self.prev_input)
                    temp_action = np.random.choice(ACTION_SPACE_SIZE, p=policy_probs)

                    # CRBP-style stopping condition:
                    # - if r>0: we want to REPEAT the previous action
                    # - if r<0: we want to AVOID repeating the previous action
                    if (r > 0 and temp_action == self.prev_action_idx) or \
                       (r < 0 and temp_action != self.prev_action_idx):
                        break  # stop rehearsal early once behavior is "fixed"

        # --- ACTION STEP ---
        policy_probs, _, _, _ = self.compute_action_policy(current_input)
        action_idx = np.random.choice(ACTION_SPACE_SIZE, p=policy_probs)

        # Store state for next learning step
        self.prev_input = current_input
        self.prev_action_idx = action_idx
        self.prev_eval = current_eval
        self.just_born = False

        # Interact with world (same physics as ANN ERL)
        self.perform_action(world, action_idx)
        
    def learn(self, state, action_idx, r_sign):
        """
        One policy-gradient-style update on the programmatic policy.

        r_sign: +1 if the last eval change was positive, -1 if negative.
        """
        policy_probs, clause_probs, action_probs_per_clause, gate_activations = self.compute_action_policy(state)

        pi_a = policy_probs[action_idx]
        if pi_a < 1e-10:
            pi_a = 1e-10

        # Clause responsibilities for the chosen action
        rho = (clause_probs * action_probs_per_clause[:, action_idx]) / pi_a

        # 1. Update Logits (Action selection within each clause)
        for i in range(K_CLAUSES):
            Si = action_probs_per_clause[i]
            grad_Li = -rho[i] * Si
            grad_Li[action_idx] += rho[i]

            # Use r_sign instead of raw r/advantage to mimic CRBP
            self.genome.action_clauses[i]['logits'] += self.alpha * r_sign * grad_Li
            # Stability clipping (you already had this)
            np.clip(self.genome.action_clauses[i]['logits'], -10, 10,
                    out=self.genome.action_clauses[i]['logits'])

        # 2. Update Gates (Structure)
        # suffix_sum_rho[k] = sum_{j>k} rho[j]
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

            # Structural dampening (0.1) as you already had
            self.genome.action_clauses[i]['w']   += self.alpha * 0.1 * r_sign * grad_w
            self.genome.action_clauses[i]['tau'] += self.alpha * 0.1 * r_sign * grad_tau    

    def perform_action(self, world, action_idx):
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
            elif isinstance(occ, Entity) and occ.type_id == TYPE_AGENT:
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
            mate = world.find_closest_mate(self)
            if mate and isinstance(mate, ProgrammaticERLAgent):
                child_genome = self.genome.crossover(mate.genome)
            else:
                # Clone
                child_genome = ProgrammaticGenome()
                child_genome.action_clauses = [copy_clause(c) for c in self.genome.action_clauses]
                child_genome.eval_clauses = [copy_clause(c) for c in self.genome.eval_clauses]
                child_genome.eval_bias = self.genome.eval_bias
            
            # Evolution strategy: Gaussian noise (standard for Programmatic/Float genomes)
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
        
        for x in range(WORLD_WIDTH):
            self.add_entity(Wall(x, 0))
            self.add_entity(Wall(x, WORLD_HEIGHT-1))
        for y in range(1, WORLD_HEIGHT-1):
            self.add_entity(Wall(0, y))
            self.add_entity(Wall(WORLD_WIDTH-1, y))
        for _ in range(50):
            self.add_entity(Wall(random.randint(1,98), random.randint(1,98)))
        for _ in range(30):
            self.add_entity(Tree(random.randint(1,98), random.randint(1,98)))
        for _ in range(200):
            self.add_entity(Plant(random.randint(1,98), random.randint(1,98)))
        for _ in range(100):
            self.spawn_agent_near(random.randint(1,98), random.randint(1,98), ProgrammaticGenome())
        for _ in range(5):
            self.add_entity(Carnivore(random.randint(1,98), random.randint(1,98)))

    def spawn_agent_near(self, x, y, genome):
        dirs = list(DIRS)
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.add_entity(ProgrammaticERLAgent(nx, ny, genome)):
                return
        self.add_entity(ProgrammaticERLAgent(random.randint(1,98), random.randint(1,98), genome))

# Helper to get metrics from Programmatic Agents
def get_population_metrics(world):
    agents = world.agents
    if not agents: return None
    # Programmatic Metric Collection
    # We average the L1 norm of weights in the Plant/Carnivore indices
    PLANT_INDICES = [2, 8, 14, 20]
    CARNIVORE_INDICES = [4, 10, 16, 22]
    
    plant_eval_vals = []
    plant_action_vals = []
    carnivore_eval_vals = []
    carnivore_action_vals = []
    
    for agent in agents:
        pe_sum, ce_sum, pa_sum, ca_sum = 0, 0, 0, 0
        for clause in agent.genome.eval_clauses:
            pe_sum += np.mean([abs(clause['w'][i]) for i in PLANT_INDICES])
            ce_sum += np.mean([abs(clause['w'][i]) for i in CARNIVORE_INDICES])
        for clause in agent.genome.action_clauses:
            pa_sum += np.mean([abs(clause['w'][i]) for i in PLANT_INDICES])
            ca_sum += np.mean([abs(clause['w'][i]) for i in CARNIVORE_INDICES])
            
        plant_eval_vals.append(pe_sum / K_CLAUSES)
        carnivore_eval_vals.append(ce_sum / K_CLAUSES)
        plant_action_vals.append(pa_sum / K_CLAUSES)
        carnivore_action_vals.append(ca_sum / K_CLAUSES)
            
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
    history = {'steps': [], 'plant_eval': [], 'plant_action': [], 'carnivore_eval': [], 'carnivore_action': []}
    
    try:
        for t in range(1, max_steps + 1):
            world.update()
            if step_callback: step_callback(t, world)
            if t % 100 == 0:
                metrics = get_population_metrics(world)
                if metrics:
                    history['steps'].append(t)
                    for k, v in metrics.items(): history[k].append(v)
            if visualize and t % 10 == 0: vis.update(world)
            if t % 10000 == 0 and not visualize: print(f"[{strategy}] Step {t}...", flush=True)
            if t % 100 == 0 and visualize:
                avg_fitness = 0
                if world.agents: avg_fitness = sum(a.energy for a in world.agents) / len(world.agents)
                print(f"Step {t}: Agents={len(world.agents)} Carnivores={len(world.carnivores)} Plants={len(world.plants)} AvgEnergy={avg_fitness:.1f}")
            if len(world.agents) == 0:
                if visualize: print("Extinction event.")
                return t, history
        return max_steps, history
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        return t, history
    finally:
        if vis: vis.close()

if __name__ == "__main__":
    run_simulation()