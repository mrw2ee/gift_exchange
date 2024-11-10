import unittest
import numpy as np
from magichat import MagicHat, map_receivers

class TestMagicHat(unittest.TestCase):
    """ Test Class MagicHat """

    def test_inputs(self):
        hat = MagicHat()
        with self.assertRaises(TypeError):
            hat.add_name(3)
        with self.assertRaises(TypeError):
            hat.add_name(["fun","b"])
        with self.assertRaises(TypeError):
            hat.add_group(["fun","b",4])

    def test_duplicate_name(self):
        # Duplicate names should not be added to list, raise warning instead
        hat = MagicHat()

        # No names yet
        self.assertEqual(len(hat.names), 0)
        # Add one name
        hat.add_name("Mike")
        self.assertEqual(len(hat.names), 1)

        # Repeated name should raise warning
        with self.assertWarns(UserWarning):
            hat.add_name("Mike")
        # ... and not added to name list
        self.assertEqual(len(hat.names), 1)

class TestMapReceivers(unittest.TestCase):
    """ Test function map_receivers """
    
    def test_inputs(self):
        with self.assertRaises(TypeError):
            # wrong dtype
            map_receivers(np.zeros(3,3))
        with self.assertRaises(AttributeError):
            # right type, not square
            map_receivers(np.array([[False,True,False],[True,True,False]],dtype=bool))

if __name__ == '__main__':
    unittest.main()