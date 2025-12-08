import unittest
import numpy as np
import random
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ERL import Genome, ERLAgent, World, Entity, Plant, Tree, Wall, Carnivore, bits_to_weight, get_distance_value
from ERL import INPUT_SIZE, ACTION_OUTPUT_SIZE, EVAL_OUTPUT_SIZE, GENOME_LENGTH, WORLD_WIDTH, WORLD_HEIGHT

class TestERLHelpers(unittest.TestCase):
    def test_bits_to_weight(self):
        # 0000 -> 0 -> (0 - 7.5)/75 = -0.1
        self.assertAlmostEqual(bits_to_weight('0000'), -0.1)
        # 1111 -> 15 -> (15 - 7.5)/75 = 0.1
        self.assertAlmostEqual(bits_to_weight('1111'), 0.1)

    def test_get_distance_value(self):
        # dist > max -> 0
        self.assertEqual(get_distance_value(5, 4), 0.0)
        # dist 1 -> 1.0
        self.assertEqual(get_distance_value(1, 4), 1.0)
        # dist max -> 0.5
        self.assertEqual(get_distance_value(4, 4), 0.5)

class TestGenome(unittest.TestCase):
    def test_init_random(self):
        g = Genome()
        self.assertEqual(len(g.bits), GENOME_LENGTH)
        
    def test_init_bits(self):
        bits = '1' * GENOME_LENGTH
        g = Genome(bits)
        self.assertEqual(g.bits, bits)
        
    def test_decode(self):
        g = Genome()
        w_action, w_eval = g.decode()
        self.assertEqual(w_action.shape, (INPUT_SIZE, ACTION_OUTPUT_SIZE))
        self.assertEqual(w_eval.shape, (INPUT_SIZE, EVAL_OUTPUT_SIZE))
        
    def test_mutate(self):
        g = Genome('0' * GENOME_LENGTH)
        # Force mutation with high rate
        g.mutate(rate=1.0)
        self.assertEqual(g.bits, '1' * GENOME_LENGTH)
        
    def test_crossover(self):
        g1 = Genome('0' * GENOME_LENGTH)
        g2 = Genome('1' * GENOME_LENGTH)
        child = g1.crossover(g2)
        self.assertEqual(len(child.bits), GENOME_LENGTH)
        # Should contain parts of both (unless random points are ends, unlikely but possible)
        # We just check length and type mainly

class TestEntities(unittest.TestCase):
    def test_entity_init(self):
        e = Entity(10, 20, 99)
        self.assertEqual(e.x, 10)
        self.assertEqual(e.y, 20)
        self.assertEqual(e.type_id, 99)
        self.assertFalse(e.dead)
        
    def test_plant_init(self):
        p = Plant(5, 5)
        self.assertEqual(p.type_id, 2)
        
    def test_tree_init(self):
        t = Tree(1, 1)
        self.assertEqual(t.type_id, 3)
        self.assertIsNone(t.occupant)
        
    def test_carnivore_init(self):
        c = Carnivore(2, 2)
        self.assertEqual(c.type_id, 4)
        self.assertGreater(c.energy, 0)
        
class TestWorld(unittest.TestCase):
    def setUp(self):
        self.world = World()
        # Clear random spawns for deterministic testing if needed, 
        # but init_world does a lot. We can just test the state.
        
    def test_init_world(self):
        self.assertEqual(len(self.world.grid), WORLD_WIDTH)
        self.assertEqual(len(self.world.grid[0]), WORLD_HEIGHT)
        # Check walls
        self.assertIsInstance(self.world.grid[0][0], Wall)
        
    def test_add_remove_entity(self):
        p = Plant(50, 50)
        # Ensure spot is empty first (might be occupied by random init)
        self.world.grid[50][50] = None 
        
        self.assertTrue(self.world.add_entity(p))
        self.assertIn(p, self.world.plants)
        self.assertEqual(self.world.grid[50][50], p)
        
        self.world.remove_entity(p)
        self.assertNotIn(p, self.world.plants)
        self.assertIsNone(self.world.grid[50][50])
        
    def test_move_entity(self):
        a = ERLAgent(10, 10)
        self.world.grid[10][10] = None # Clear spot
        self.world.add_entity(a)
        
        self.world.grid[11][10] = None # Clear target
        self.world.move_entity(a, 11, 10)
        
        self.assertEqual(a.x, 11)
        self.assertEqual(a.y, 10)
        self.assertIsNone(self.world.grid[10][10])
        self.assertEqual(self.world.grid[11][10], a)

class TestERLAgent(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.agent = ERLAgent(50, 50)
        self.world.grid[50][50] = self.agent
        
    def test_get_inputs(self):
        inputs = self.agent.get_inputs(self.world)
        self.assertEqual(len(inputs), INPUT_SIZE)
        
    def test_perform_action_move(self):
        # Force empty neighbor
        self.world.grid[50][49] = None # North
        
        # Action 0 is North (0, -1)
        self.agent.perform_action(self.world, 0)
        
        self.assertEqual(self.agent.x, 50)
        self.assertEqual(self.agent.y, 49)
        
    def test_perform_action_eat_plant(self):
        # Prevent reproduction for this test
        import ERL
        original_threshold = ERL.REPRODUCE_ENERGY_THRESHOLD
        ERL.REPRODUCE_ENERGY_THRESHOLD = 200.0
        
        try:
            # Ensure target is clear of random walls
            self.world.grid[50][49] = None
            p = Plant(50, 49)
            self.world.grid[50][49] = p
            self.world.plants.append(p)
            
            initial_energy = self.agent.energy
            self.agent.perform_action(self.world, 0) # Move North
            
            self.assertGreater(self.agent.energy, initial_energy)
            self.assertNotIn(p, self.world.plants)
        finally:
            ERL.REPRODUCE_ENERGY_THRESHOLD = original_threshold
        
    def test_perform_action_hit_wall(self):
        w = Wall(50, 49)
        self.world.grid[50][49] = w
        
        initial_health = self.agent.health
        self.agent.perform_action(self.world, 0) # Move North
        
        self.assertLess(self.agent.health, initial_health)
        self.assertEqual(self.agent.x, 50) # Did not move
        self.assertEqual(self.agent.y, 50)

if __name__ == '__main__':
    unittest.main()
