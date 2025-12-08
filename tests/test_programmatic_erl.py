import unittest
import numpy as np
import random
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from programmatic_erl import ProgrammaticGenome, ProgrammaticERLAgent, ProgrammaticWorld, K_CLAUSES, ACTION_SPACE_SIZE, INPUT_DIM
from ERL import INPUT_SIZE, DIRS

class TestProgrammaticGenome(unittest.TestCase):
    def test_init(self):
        g = ProgrammaticGenome()
        self.assertEqual(len(g.action_clauses), K_CLAUSES)
        self.assertEqual(len(g.eval_clauses), K_CLAUSES)
        
        # Check structure of a clause
        clause = g.action_clauses[0]
        self.assertIn('w', clause)
        self.assertIn('tau', clause)
        self.assertIn('logits', clause)
        self.assertEqual(clause['w'].shape, (INPUT_DIM,))
        self.assertEqual(clause['logits'].shape, (ACTION_SPACE_SIZE,))
        
    def test_mutate(self):
        g = ProgrammaticGenome()
        original_w = g.action_clauses[0]['w'].copy()
        
        # Force mutation
        random.seed(42)
        np.random.seed(42)
        g.mutate(rate=1.0, perturbation_std=0.1)
        
        self.assertFalse(np.array_equal(g.action_clauses[0]['w'], original_w))
        
    def test_crossover(self):
        g1 = ProgrammaticGenome()
        g2 = ProgrammaticGenome()
        child = g1.crossover(g2)
        
        self.assertIsInstance(child, ProgrammaticGenome)
        self.assertEqual(len(child.action_clauses), K_CLAUSES)

class TestProgrammaticERLAgent(unittest.TestCase):
    def setUp(self):
        self.world = ProgrammaticWorld()
        self.agent = ProgrammaticERLAgent(50, 50)
        self.world.grid[50][50] = self.agent
        
    def test_compute_action_policy(self):
        state = np.zeros(INPUT_DIM)
        policy_probs, clause_probs, action_probs_per_clause, gate_activations = self.agent.compute_action_policy(state)
        
        self.assertEqual(policy_probs.shape, (ACTION_SPACE_SIZE,))
        self.assertAlmostEqual(np.sum(policy_probs), 1.0)
        self.assertEqual(clause_probs.shape, (K_CLAUSES,))
        self.assertEqual(action_probs_per_clause.shape, (K_CLAUSES, ACTION_SPACE_SIZE))
        
    def test_compute_evaluation(self):
        state = np.zeros(INPUT_DIM)
        eval_val = self.agent.compute_evaluation(state)
        self.assertTrue(0.0 <= eval_val <= 1.0)
        
    def test_learn(self):
        state = np.zeros(INPUT_DIM)
        action_idx = 0
        r_sign = 1.0
        
        # Capture initial weights
        initial_w = self.agent.genome.action_clauses[0]['w'].copy()
        initial_logits = self.agent.genome.action_clauses[0]['logits'].copy()
        
        self.agent.learn(state, action_idx, r_sign)
        
        # Weights should change (unless gradients are exactly zero, which is unlikely with random init)
        # Note: If state is all zeros, grad_w might be zero. Let's set some input.
        state[0] = 1.0
        self.agent.learn(state, action_idx, r_sign)
        
        # Check if logits changed (they depend on rho, not state directly)
        self.assertFalse(np.array_equal(self.agent.genome.action_clauses[0]['logits'], initial_logits))

class TestProgrammaticWorld(unittest.TestCase):
    def test_init(self):
        w = ProgrammaticWorld()
        self.assertEqual(len(w.agents), 100)
        self.assertIsInstance(w.agents[0], ProgrammaticERLAgent)

if __name__ == '__main__':
    unittest.main()
