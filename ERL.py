import random
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- CONFIGURATION & CONSTANTS ---

WORLD_WIDTH = 100
WORLD_HEIGHT = 100

# Agent/Network Constants
INPUT_SIZE = 28   # (6 types * 4 directions) + 4 internal (health, energy, intree, bias)
HIDDEN_SIZE = 0   # Single layer as per paper
ACTION_OUTPUT_SIZE = 2 # 2 bits for N, S, E, W
EVAL_OUTPUT_SIZE = 1
TOTAL_WEIGHTS = 84 # (28 * 2 action) + (28 * 1 eval) = 84. Matches paper exactly.

GENOME_BITS_PER_WEIGHT = 4
GENOME_LENGTH = TOTAL_WEIGHTS * GENOME_BITS_PER_WEIGHT

# Learning Constants (CRBP)
LEARNING_RATE_POS = 0.05 # Fast learning to fix broken weights
LEARNING_RATE_NEG = 0.05 # Fast learning to fix broken weights
NOISE_PROB = 0.5       # Balanced

# Simulation Constants
AGENT_VIEW_DIST = 4 # Increased to 20 to give advantage to smart agents
CARNIVORE_VIEW_DIST = 6 # Paper: 6 cells
CARNIVORE_SPAWN_FREQ = 200

# Strategy Configuration
# ERL: Evolution + Learning
# E: Evolution onlyx
# L: Learning only (No inheritance)
# F: Fixed (No evolution, no learning)
# B: Brownian (Random walk)
SIM_STRATEGY = 'ERL'

# Energy/Health Dynamics (Inferred standard AL values)
MAX_ENERGY = 100.0
MAX_HEALTH = 1.0
ENERGY_COST_MOVE = 0.3 # Very High cost
ENERGY_GAIN_PLANT = 100.0 # Huge reward
ENERGY_GAIN_MEAT = 50.0
DAMAGE_CARNIVORE = 0.0 # Harmless
DAMAGE_WALL = 0.1 # Survivable
REPRODUCE_ENERGY_THRESHOLD = 60.0 # Easier to reproduce
REPRODUCE_COST = 50.0 # Lower cost
PLANT_GROWTH_PROB = 0.005 # Very scarce food
PLANT_MAX_DENSITY_NEIGHBORS = 4
TREE_BIRTH_PROB = 0.001 # Infrequent
TREE_DEATH_PROB = 0.001 # Infrequent

# Object Types IDs
TYPE_EMPTY = 0
TYPE_WALL = 1
TYPE_PLANT = 2
TYPE_TREE = 3
TYPE_CARNIVORE = 4
TYPE_AGENT = 5

# Directions: N, E, S, W
# Directions: N, E, W, S
# Reordered so that complements are opposites:
# 0 (00) -> N, 3 (11) -> S
# 1 (01) -> E, 2 (10) -> W
DIRS = [(0, -1), (1, 0), (-1, 0), (0, 1)] # x, y changes

# --- HELPER FUNCTIONS ---

def sigmoid(x):
    # Fastest version: 1 / (1 + exp(-x))
    # Overflow in exp(-x) yields inf, and 1/(1+inf) -> 0.0, which is correct for sigmoid(large_negative).
    # We suppress warnings globally or just accept them.
    return 1.0 / (1.0 + np.exp(-x))

def bits_to_weight(bits):
    """Decodes 4 bits into a weight value.
    Mapping 0-15 integer to range roughly -4.0 to +4.0"""
    val = int(bits, 2)
    # Map 0..15 to -0.1 .. +0.1
    # Small weights = Sigmoid closer to 0.5 = Random initial behavior (like Brownian)
    return (val - 7.5) / 75.0

def get_distance_value(dist, max_dist):
    """Paper: 'value from 0.5 to 1.0 proportional to closeness'"""
    if dist > max_dist: return 0.0 
    # dist 1 -> 1.0, dist max -> 0.5
    return 1.0 - (0.5 * (dist - 1) / (max_dist - 1)) if max_dist > 1 else 1.0

# --- CLASSES ---

class Genome:
    def __init__(self, bits=None):
        if bits:
            self.bits = bits
        else:
            self.bits = "".join([str(random.randint(0, 1)) for _ in range(GENOME_LENGTH)])
    
    def decode(self):
        weights = []
        for i in range(0, GENOME_LENGTH, GENOME_BITS_PER_WEIGHT):
            chunk = self.bits[i:i+GENOME_BITS_PER_WEIGHT]
            weights.append(bits_to_weight(chunk))
        
        # Split into Action weights and Eval weights
        # Action: 28 inputs * 2 outputs = 56 weights
        # Eval: 28 inputs * 1 output = 28 weights
        action_w = np.array(weights[:56]).reshape(INPUT_SIZE, ACTION_OUTPUT_SIZE)
        eval_w = np.array(weights[56:]).reshape(INPUT_SIZE, EVAL_OUTPUT_SIZE)
        return action_w, eval_w

    def mutate(self, rate=0.05): # Increased to 0.05 to break E
        new_bits = list(self.bits)
        for i in range(len(new_bits)):
            if random.random() < rate:
                new_bits[i] = '1' if new_bits[i] == '0' else '0'
        self.bits = "".join(new_bits)

    def crossover(self, other):
        # Two-point crossover
        p1 = random.randint(0, GENOME_LENGTH - 1)
        p2 = random.randint(0, GENOME_LENGTH - 1)
        start, end = min(p1, p2), max(p1, p2)
        
        child_bits = self.bits[:start] + other.bits[start:end] + self.bits[end:]
        return Genome(child_bits)

class Entity:
    def __init__(self, x, y, type_id):
        self.x = x
        self.y = y
        self.type_id = type_id
        self.dead = False

class Plant(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, TYPE_PLANT)

class Tree(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, TYPE_TREE)
        self.occupant = None

class Wall(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, TYPE_WALL)

class Carnivore(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, TYPE_CARNIVORE)
        self.energy = MAX_ENERGY * 0.5
        self.health = MAX_HEALTH
        
    def step(self, world):
        # 1. Perception (Closest agent within 6 cells)
        target_dir = None
        min_dist = CARNIVORE_VIEW_DIST + 1
        
        # Scan for closest agent
        # Simply iterating all agents for efficiency instead of raytracing grid
        # in a real 100x100 grid, raytracing is better, but here agents list is small
        closest_agent = None
        for agent in world.agents:
            if agent.in_tree: continue # Can't see/target agents in trees
            d = abs(agent.x - self.x) + abs(agent.y - self.y) # Manhattan dist
            if d <= CARNIVORE_VIEW_DIST and d < min_dist:
                min_dist = d
                closest_agent = agent
        
        if closest_agent:
            # Determine direction N, S, E, W
            dx = closest_agent.x - self.x
            dy = closest_agent.y - self.y
            
            # Simple FSA movement preference
            # Simple FSA movement preference
            if abs(dx) > abs(dy):
                target_dir = 1 if dx > 0 else 2 # E or W
            else:
                target_dir = 3 if dy > 0 else 0 # S or N
        else:
            # Random walk
            target_dir = random.randint(0, 3)
            
        # Move Logic
        dx, dy = DIRS[target_dir]
        nx, ny = self.x + dx, self.y + dy
        
        # Check bounds
        if 0 <= nx < WORLD_WIDTH and 0 <= ny < WORLD_HEIGHT:
            # Interaction
            occ = world.get_occupant(nx, ny)
            if occ is None or isinstance(occ, Plant):
                # Move
                world.move_entity(self, nx, ny)
                self.energy -= ENERGY_COST_MOVE
            elif isinstance(occ, ERLAgent): # Changed from Agent to ERLAgent
                # Damage Agent
                occ.health -= DAMAGE_CARNIVORE
                occ.last_damage_source = 'carnivore'
                # Eat dead agent logic handled in global loop or here?
                # Paper: "Eat dead agents". If agent dies, carnivore gains energy.
                if occ.health <= 0:
                    self.energy += ENERGY_GAIN_MEAT
                # Carnivore doesn't move onto living agent, just hits it
            elif isinstance(occ, Wall):
                pass
            elif isinstance(occ, Tree):
                pass 
                
        # Reproduction
        if self.energy > REPRODUCE_ENERGY_THRESHOLD:
            self.energy -= REPRODUCE_COST
            # Spawn new carnivore nearby
            world.spawn_carnivore_near(self.x, self.y)
            
        # Death
        if self.energy <= 0 or self.health <= 0:
            self.dead = True

class ERLAgent(Entity):
    def __init__(self, x, y, genome=None):
        super().__init__(x, y, TYPE_AGENT)
        self.genome = genome if genome else Genome()
        
        # Decode weights
        # W_action: 28x2, W_eval: 28x1
        self.w_action, self.w_eval = self.genome.decode()
        
        self.energy = MAX_ENERGY * 0.5
        self.health = MAX_HEALTH
        self.in_tree = False
        self.age = 0
        
        # State for learning
        self.prev_input = np.zeros(INPUT_SIZE)
        self.prev_action_probs = np.zeros(ACTION_OUTPUT_SIZE)
        self.prev_action_idx = 0
        self.prev_eval = 0.0
        self.prev_energy = self.energy
        self.prev_health = self.health # Track previous health
        self.just_born = True
        
        # Pre-allocated buffers for optimization
        self.grad_buffer = np.zeros((INPUT_SIZE, ACTION_OUTPUT_SIZE))
        self.errors_buffer = np.zeros(ACTION_OUTPUT_SIZE)
        
    def get_inputs(self, world):
        # 28 Inputs:
        # [Wall, Plant, Tree, Carnivore, Agent, Empty] for N, E, S, W (24 inputs)
        # [Health, Energy, InTree, Bias] (4 inputs)
        
        inputs = np.zeros(INPUT_SIZE)
        grid = world.grid # Direct access
        
        # Visual
        for i, (dx, dy) in enumerate(DIRS): # N, E, S, W
            closest_type = TYPE_EMPTY
            closest_dist = AGENT_VIEW_DIST + 1
            
            # Ray cast
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
            
            # Map type to input index
            # Types: Empty(0), Wall(1), Plant(2), Tree(3), Carnivore(4), Agent(5)
            # 6 types per direction
            base_idx = i * 6
            type_idx = base_idx + closest_type
            inputs[type_idx] = val
            
        # Proprioceptors
        offset = 24
        inputs[offset] = self.health
        inputs[offset+1] = self.energy / MAX_ENERGY
        inputs[offset+2] = 1.0 if self.in_tree else 0.0
        inputs[offset+3] = 1.0 # Bias
        
        return inputs

    def step(self, world):
        if SIM_STRATEGY == 'B':
            # Brownian motion - random walk
            self.age += 1
            action_idx = random.randint(0, 3)
            self.perform_action(world, action_idx)
            
            # Reproduction logic for Brownian (still needs to reproduce to survive)
            self.check_reproduction(world)
            
            # Death check
            if self.energy <= 0 or self.health <= 0:
                self.dead = True
            return

        self.age += 1
        current_input = self.get_inputs(world)
        
        # --- L1: Evaluation ---
        # Eval Net Forward: Input (1x28) * Weights (28x1) -> Scalar
        # Inline sigmoid: 1 / (1 + exp(-x))
        dot_eval = np.dot(current_input, self.w_eval)[0]
        current_eval = 1.0 / (1.0 + np.exp(-dot_eval))
        
        # --- L2: Learning (CRBP) ---
        # ERL and L learn. E, F, B do not.
        if SIM_STRATEGY in ['ERL', 'L'] and not self.just_born:
            # Reinforcement Signal (Step 1)
            r = current_eval - self.prev_eval
            pass
            
            # Step 1: If r = 0, go to 6 (Skip learning)
            if r != 0:
                # Reconstruct previous action (o)
                # Paper says "Output is 2 bits coding action direction N, S, E, W".
                action_bits = [(self.prev_action_idx >> 1) & 1, self.prev_action_idx & 1]
                
                # Mental Rehearsal Loop (Step 5)
                # We loop until the network output matches the desired outcome (or max iterations)
                for _ in range(20): # Safety limit for loop
                    # Forward pass on previous input (s_j)
                    # Inline sigmoid
                    dot_action = np.dot(self.prev_input, self.w_action)
                    net_out = 1.0 / (1.0 + np.exp(-dot_action))
                    
                    # Error Calculation (Step 2)
                    # Unrolled for ACTION_OUTPUT_SIZE = 2
                    s0 = net_out[0]
                    s1 = net_out[1]
                    o0 = action_bits[0]
                    o1 = action_bits[1]
                    
                    d0 = s0 * (1.0 - s0)
                    d1 = s1 * (1.0 - s1)
                    
                    if r > 0:
                        e0 = (o0 - s0) * d0
                        e1 = (o1 - s1) * d1
                    else:
                        e0 = ((1.0 - o0) - s0) * d0
                        e1 = ((1.0 - o1) - s1) * d1
                        
                    self.errors_buffer[0] = e0
                    self.errors_buffer[1] = e1
                    
                    # Update Weights (Step 4)
                    learning_rate = LEARNING_RATE_POS if r > 0 else LEARNING_RATE_NEG
                    
                    # Zero-allocation update:
                    # 1. Compute gradient into buffer
                    np.multiply(self.prev_input[:, None], self.errors_buffer[None, :], out=self.grad_buffer)
                    
                    # 2. Scale by learning rate (in-place)
                    self.grad_buffer *= learning_rate
                    
                    # 3. Update weights (in-place)
                    self.w_action += self.grad_buffer
                    
                    # Step 5: Forward propagate again to produce new s_j's
                    # Inline sigmoid
                    dot_new = np.dot(self.prev_input, self.w_action)
                    new_net_out = 1.0 / (1.0 + np.exp(-dot_new))
                    
                    # Generate temporary output vector o*
                    # Unrolled bit generation
                    s_j0 = new_net_out[0]
                    prob0 = (s_j0 - 0.5) / NOISE_PROB + 0.5
                    bit0 = 1 if prob0 >= random.random() else 0
                    
                    s_j1 = new_net_out[1]
                    prob1 = (s_j1 - 0.5) / NOISE_PROB + 0.5
                    bit1 = 1 if prob1 >= random.random() else 0
                    
                    # Check condition: If (r > 0 and o* != o) or (r < 0 and o* == o), go to 2
                    o_matches = (bit0 == o0 and bit1 == o1)
                    
                    if (r > 0 and not o_matches) or (r < 0 and o_matches):
                        continue # Loop again
                    else:
                        break # Condition satisfied, exit loop
            
        
        # --- L3: Behave ---
        # Action Net Forward
        net_out = sigmoid(np.dot(current_input, self.w_action))
        
        # Stochastic Output Generation (Step 7 in CRBP Fig 3)
        # "If (s_j - 0.5)/v + 0.5 >= random(0,1) then 1 else 0"
        
        output_bits = []
        for j in range(ACTION_OUTPUT_SIZE):
            s_j = net_out[j]
            prob = (s_j - 0.5) / NOISE_PROB + 0.5
            bit = 1 if prob >= random.random() else 0
            output_bits.append(bit)
            
        # Decode bits to action
        action_idx = (output_bits[0] << 1) | output_bits[1] # 0, 1, 2, 3
        
        # Store state for next step (Energy/Health before action)
        self.prev_energy = self.energy
        self.prev_health = self.health

        # Execute Action
        self.perform_action(world, action_idx)
        
        # Store state for next step
        self.prev_input = current_input
        self.prev_action_idx = action_idx
        self.prev_eval = current_eval
        self.just_born = False
        
    def perform_action(self, world, action_idx):
        dx, dy = DIRS[action_idx]
        nx, ny = self.x + dx, self.y + dy
        
        # Consumption/Movement Logic
        if 0 <= nx < WORLD_WIDTH and 0 <= ny < WORLD_HEIGHT:
            occ = world.get_occupant(nx, ny)
            
            if occ is None:
                # Move
                if self.in_tree: self.in_tree = False # Climbed down
                world.move_entity(self, nx, ny)
                self.energy -= ENERGY_COST_MOVE
                
            elif isinstance(occ, Plant):
                # Eat Plant
                world.remove_entity(occ)
                self.energy += ENERGY_GAIN_PLANT
                if self.energy > MAX_ENERGY: self.energy = MAX_ENERGY
                # Move into spot
                world.move_entity(self, nx, ny)
                
            elif isinstance(occ, Tree):
                # Climb Tree
                if occ.occupant is None:
                    # Move to tree coords but mark as in_tree
                    # Need to handle logical position
                    world.move_entity(self, nx, ny)
                    self.in_tree = True
                    occ.occupant = self
                else:
                    # Tree occupied
                    self.energy -= ENERGY_COST_MOVE
                    
            elif isinstance(occ, Wall):
                # Damage
                self.health -= DAMAGE_WALL
                self.energy -= ENERGY_COST_MOVE
                
            elif isinstance(occ, Carnivore):
                if occ.dead:
                    # Eat dead carnivore
                    world.remove_entity(occ)
                    self.energy += ENERGY_GAIN_MEAT
                else:
                    # Damage living carnivore
                    occ.health -= DAMAGE_CARNIVORE # Agent damages carnivore
                    self.energy -= ENERGY_COST_MOVE
                
            elif isinstance(occ, ERLAgent):
                if occ.dead:
                    # Eat dead agent
                    world.remove_entity(occ)
                    self.energy += ENERGY_GAIN_MEAT
                else:
                    # Damage living agent
                    occ.health -= DAMAGE_CARNIVORE # Treat same as carnivore damage? Table says "Damage other"
                    occ.last_damage_source = 'agent'
                    self.energy -= ENERGY_COST_MOVE
        else:
            # Hit boundary wall
            self.health -= DAMAGE_WALL
            self.energy -= ENERGY_COST_MOVE
            
        # Reproduction check
        self.check_reproduction(world)
            
    def check_reproduction(self, world):
        if self.energy > REPRODUCE_ENERGY_THRESHOLD:
            child_genome = None
            
            if SIM_STRATEGY in ['L', 'F', 'B']:
                # No inheritance - random genome
                child_genome = Genome()
            else:
                # Evolution (ERL, E) - Inheritance
                mate = world.find_closest_mate(self)
                if mate:
                    child_genome = self.genome.crossover(mate.genome)
                else:
                    child_genome = Genome(self.genome.bits)
                child_genome.mutate()
            
            self.energy -= REPRODUCE_COST
            world.spawn_agent_near(self.x, self.y, child_genome)
            
        # Death
        if self.energy <= 0 or self.health <= 0:
            self.dead = True


class World:
    def __init__(self):
        self.grid = [[None for _ in range(WORLD_HEIGHT)] for _ in range(WORLD_WIDTH)]
        self.plants = []
        self.trees = []
        self.walls = []
        self.carnivores = []
        self.agents = []
        self.step_count = 0
        
        self.init_world()
        
    def init_world(self):
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
            self.spawn_agent_near(random.randint(1,98), random.randint(1,98), Genome())
            
        # Initial Carnivores
        for _ in range(5):
            self.add_entity(Carnivore(random.randint(1,98), random.randint(1,98)))

    def add_entity(self, entity):
        if 0 <= entity.x < WORLD_WIDTH and 0 <= entity.y < WORLD_HEIGHT:
            if self.grid[entity.x][entity.y] is None:
                self.grid[entity.x][entity.y] = entity
                if isinstance(entity, Plant): self.plants.append(entity)
                elif isinstance(entity, Tree): self.trees.append(entity)
                elif isinstance(entity, Wall): self.walls.append(entity)
                elif isinstance(entity, Carnivore): self.carnivores.append(entity)
                elif isinstance(entity, ERLAgent): self.agents.append(entity) # Changed from Agent to ERLAgent
                return True
        return False

    def remove_entity(self, entity):
        if self.grid[entity.x][entity.y] == entity:
            self.grid[entity.x][entity.y] = None
        
        if isinstance(entity, Plant) and entity in self.plants: self.plants.remove(entity)
        elif isinstance(entity, Carnivore) and entity in self.carnivores: self.carnivores.remove(entity)
        elif isinstance(entity, ERLAgent) and entity in self.agents: self.agents.remove(entity) # Changed from Agent to ERLAgent
        # Trees/Walls usually permanent

    def move_entity(self, entity, nx, ny):
        if self.grid[entity.x][entity.y] == entity:
            self.grid[entity.x][entity.y] = None
        entity.x, entity.y = nx, ny
        self.grid[nx][ny] = entity

    def get_occupant(self, x, y):
        return self.grid[x][y]

    def spawn_agent_near(self, x, y, genome):
        # Try adjacent cells
        dirs = list(DIRS)
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.add_entity(ERLAgent(nx, ny, genome)):
                return
        # If crowded, random spot
        self.add_entity(ERLAgent(random.randint(1,98), random.randint(1,98), genome))

    def spawn_carnivore_near(self, x, y):
        dirs = list(DIRS)
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.add_entity(Carnivore(nx, ny)):
                return

    def find_closest_mate(self, agent):
        closest = None
        min_dist = 10.0 # Mating range
        for other in self.agents:
            if other == agent: continue
            d = abs(other.x - agent.x) + abs(other.y - agent.y)
            if d < min_dist:
                min_dist = d
                closest = other
        return closest

    def update(self):
        self.step_count += 1
        
        # 1. Plants Grow
        # Geometric growth up to crowding limit
        new_plants = []
        for p in self.plants:
            if random.random() < PLANT_GROWTH_PROB:
                # Count neighbors
                neighbors = 0
                for dx, dy in DIRS:
                    if 0 <= p.x+dx < WORLD_WIDTH and 0 <= p.y+dy < WORLD_HEIGHT:
                        if isinstance(self.grid[p.x+dx][p.y+dy], Plant):
                            neighbors += 1
                
                if neighbors < PLANT_MAX_DENSITY_NEIGHBORS:
                    # Spawn new plant
                    dx, dy = random.choice(DIRS)
                    nx, ny = p.x+dx, p.y+dy
                    if 0 <= nx < WORLD_WIDTH and 0 <= ny < WORLD_HEIGHT and self.grid[nx][ny] is None:
                        new_plants.append(Plant(nx, ny))
        
        for np in new_plants:
            self.add_entity(np)
            
        # Ensure minimum plants
        if len(self.plants) < 50:
             self.add_entity(Plant(random.randint(1,98), random.randint(1,98)))

        # 2. Carnivores Act
        for c in self.carnivores[:]:
            if not c.dead:
                c.step(self)
            if c.dead:
                self.remove_entity(c)

        # Spawn new carnivore periodically
        if self.step_count % CARNIVORE_SPAWN_FREQ == 0:
            self.add_entity(Carnivore(random.randint(1,98), random.randint(1,98)))

        # 3. Agents Act
        for a in self.agents[:]:
            if not a.dead:
                a.step(self)
            if a.dead:
                # Leave corpse? Paper says "Eat dead agents". 
                # For simplicity, we convert dead agent to a temporary food object or just leave it marked dead?
                # The logic in perform_action checks if occ.dead.
                # So we leave it in the list/grid but mark flag? 
                # World.remove_entity actually removes from list.
                # We need to keep corpses for a bit.
                # Let's handle corpses: remove from 'agents' list (so they don't act) but keep in grid.
                # To simplify, we'll just remove them immediately in this basic version,
                # effectively assuming they decay instantly or are eaten instantly if collision happens same tick.
                # To be precise to paper: "Eat dead agents... decay until energy gone".
                # Implementation: Keep in grid, change type/flag, stop calling step().
                pass
        
        # Cleanup dead agents from active list
        dead_agents = [a for a in self.agents if a.dead]
        for da in dead_agents:
            # Check if we should keep corpse
            if da.energy > 0:
                # It's a corpse with meat.
                # We remove from self.agents so it doesn't think, but keep in grid.
                # We need a Corpse entity? Or just use Agent class with dead=True.
                # The loop iterates self.agents. So if we remove from list but keep in grid...
                if da in self.agents: self.agents.remove(da)
                # It stays in self.grid[x][y]
            else:
                self.remove_entity(da)



class Visualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Colors: Empty(0)=White, Wall(1)=Black, Plant(2)=Green, Tree(3)=Brown, Carnivore(4)=Red, Agent(5)=Blue
        self.cmap = ListedColormap(['white', 'black', 'green', 'brown', 'red', 'blue'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]
        self.norm = plt.Normalize(vmin=0, vmax=6)
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.img = self.ax.imshow(np.zeros((width, height)), cmap=self.cmap, norm=self.norm, interpolation='nearest')
        plt.title("ERL Simulation")
        plt.axis('off')
        
        # Legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Empty'),
            Patch(facecolor='black', edgecolor='gray', label='Wall'),
            Patch(facecolor='green', edgecolor='gray', label='Plant'),
            Patch(facecolor='brown', edgecolor='gray', label='Tree'),
            Patch(facecolor='red', edgecolor='gray', label='Carnivore'),
            Patch(facecolor='blue', edgecolor='gray', label='Agent')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        
    def update(self, world):
        # Build grid array
        # We need to map entities to IDs
        # TYPE_EMPTY = 0, TYPE_WALL = 1, TYPE_PLANT = 2, TYPE_TREE = 3, TYPE_CARNIVORE = 4, TYPE_AGENT = 5
        
        grid_data = np.zeros((self.width, self.height))
        
        # We can iterate entities or the grid. Grid is faster if populated correctly.
        # world.grid contains objects.
        
        for x in range(self.width):
            for y in range(self.height):
                obj = world.grid[x][y]
                if obj:
                    grid_data[y, x] = obj.type_id # Transpose for visual (y is row, x is col)
                else:
                    grid_data[y, x] = 0
                    
        self.img.set_data(grid_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001) # Small pause to update

    def close(self):
        plt.close(self.fig)

# Indices for inputs (N, E, S, W)
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
        # Eval weights
        pe = np.mean([agent.w_eval[i][0] for i in PLANT_INDICES])
        ce = np.mean([agent.w_eval[i][0] for i in CARNIVORE_INDICES])
        plant_eval_vals.append(pe)
        carnivore_eval_vals.append(ce)
        
        # Action weights (mean of absolute values)
        pa = np.mean([np.mean(np.abs(agent.w_action[i])) for i in PLANT_INDICES])
        ca = np.mean([np.mean(np.abs(agent.w_action[i])) for i in CARNIVORE_INDICES])
        plant_action_vals.append(pa)
        carnivore_action_vals.append(ca)
        
    return {
        'plant_eval': np.mean(plant_eval_vals),
        'plant_action': np.mean(plant_action_vals),
        'carnivore_eval': np.mean(carnivore_eval_vals),
        'carnivore_action': np.mean(carnivore_action_vals)
    }

# --- MAIN RUNNER ---

def run_simulation(strategy='ERL', visualize=True, max_steps=10000, seed=None, step_callback=None):
    global SIM_STRATEGY
    SIM_STRATEGY = strategy
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # print(f"Simulation Seed: {seed}")
    
    world = World()
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
                # Print progress even if not visualizing to debug hangs
                if not visualize:
                    print(f"[{strategy}] Step {t}...", flush=True)

            if t % 100 == 0:
                # Only print if visualizing or infrequent
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