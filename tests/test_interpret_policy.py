import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpret_policy import get_input_name, get_action_name, get_grid_snapshot, interpret_genome
from programmatic_erl import ProgrammaticWorld, ProgrammaticGenome

class TestInterpretPolicy(unittest.TestCase):
    def test_get_input_name(self):
        # 0 -> Empty (North)
        self.assertEqual(get_input_name(0), "Empty (North)")
        # 24 -> Health
        self.assertEqual(get_input_name(24), "Health")
        
    def test_get_action_name(self):
        self.assertEqual(get_action_name(0), "Move North")
        self.assertEqual(get_action_name(3), "Move South")
        
    def test_get_grid_snapshot(self):
        world = ProgrammaticWorld()
        # Just check shape and type
        snapshot = get_grid_snapshot(world)
        self.assertEqual(snapshot.shape, (100, 100))
        
    def test_interpret_genome_runs(self):
        # Just ensure it doesn't crash
        g = ProgrammaticGenome()
        # Redirect stdout to suppress print output during test
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            interpret_genome(g)
        finally:
            sys.stdout = sys.__stdout__
            
        self.assertIn("PROGRAMMATIC POLICY INTERPRETATION", captured_output.getvalue())

if __name__ == '__main__':
    unittest.main()
