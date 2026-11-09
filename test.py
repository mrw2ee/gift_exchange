
import unittest
import numpy as np
from magichat import MagicHat, map_receivers


class TestMagicHat(unittest.TestCase):
    """Test Class MagicHat"""

    def test_inputs(self):
        hat = MagicHat()
        with self.assertRaises(TypeError):
            hat.add_name(3)
        with self.assertRaises(TypeError):
            hat.add_name(["fun", "b"])
        with self.assertRaises(TypeError):
            hat.add_group(["fun", "b", 4])

    def test_duplicate_name(self):
        # Duplicate names should not be added to list, raise warning instead
        hat = MagicHat()

        self.assertEqual(len(hat.names), 0)

        hat.add_name("Mike")
        self.assertEqual(len(hat.names), 1)

        with self.assertWarns(UserWarning):
            hat.add_name("Mike")

        self.assertEqual(len(hat.names), 1)

    def test_add_group(self):
        hat = MagicHat()

        hat.add_group(["Mike", "Alice"])

        # Names should automatically be added
        self.assertEqual(len(hat.names), 2)
        self.assertIn("Mike", hat.names)
        self.assertIn("Alice", hat.names)

        # Group should be recorded
        self.assertEqual(len(hat.groups), 1)
        self.assertEqual(hat.groups[0], ["Mike", "Alice"])

    def test_mask(self):
        hat = MagicHat()

        hat.add_group(["Mike", "Alice"])
        hat.add_name("Bob")

        mask = hat.mask

        mike = hat.names.index("Mike")
        alice = hat.names.index("Alice")
        bob = hat.names.index("Bob")

        # Nobody can gift themselves
        self.assertFalse(mask[mike, mike])
        self.assertFalse(mask[alice, alice])
        self.assertFalse(mask[bob, bob])

        # Group members cannot gift each other
        self.assertFalse(mask[mike, alice])
        self.assertFalse(mask[alice, mike])

        # Other pairings are allowed
        self.assertTrue(mask[mike, bob])
        self.assertTrue(mask[bob, mike])


class TestMapReceivers(unittest.TestCase):
    """Test function map_receivers"""

    def test_inputs(self):
        with self.assertRaises(TypeError):
            # Wrong dtype
            map_receivers(np.zeros((3, 3)))

        with self.assertRaises(AttributeError):
            # Right type, not square
            map_receivers(
                np.array(
                    [[False, True, False],
                     [True, True, False]],
                    dtype=bool
                )
            )

    def test_everyone_gives_and_receives_once(self):
        hat = MagicHat()

        for name in ["Alice", "Bob", "Charlie", "David"]:
            hat.add_name(name)

        result = map_receivers(hat.mask)

        self.assertEqual(len(result), 4)

        # Everyone receives exactly once
        np.testing.assert_array_equal(
            np.sort(result), np.arange(4)
        )

    def test_no_self_gifting(self):
        hat = MagicHat()

        for name in ["Alice", "Bob", "Charlie", "David"]:
            hat.add_name(name)

        result = map_receivers(hat.mask)

        for giver, receiver in enumerate(result):
            self.assertNotEqual(giver, receiver)

    def test_group_restrictions(self):
        hat = MagicHat()

        hat.add_group(["Alice", "Bob"])
        hat.add_name("Charlie")
        hat.add_name("David")

        result = map_receivers(hat.mask)

        alice = hat.names.index("Alice")
        bob = hat.names.index("Bob")

        self.assertNotEqual(result[alice], bob)
        self.assertNotEqual(result[bob], alice)

    def test_no_reciprocal_exchanges(self):
        hat = MagicHat()

        for name in ["Alice", "Bob", "Charlie", "David", "Eve"]:
            hat.add_name(name)

        result = map_receivers(
            hat.mask,
            prevent_reciprocal=True
        )

        # No A -> B and B -> A pairings
        for giver, receiver in enumerate(result):
            self.assertNotEqual(result[receiver], giver)

    def test_three_person_cycle_allowed(self):
        # Three-person circular exchanges should be allowed
        mask = np.array([
            [False, True, True],
            [True, False, True],
            [True, True, False]
        ], dtype=bool)

        result = map_receivers(
            mask,
            prevent_reciprocal=True
        )

        # Everyone gives and receives exactly once
        np.testing.assert_array_equal(
            np.sort(result), np.arange(3)
        )

        # No reciprocal exchanges
        for giver, receiver in enumerate(result):
            self.assertNotEqual(result[receiver], giver)

    def test_impossible_assignment_raises_exception(self):
        # Two participants cannot avoid reciprocal exchanges
        mask = np.array([
            [False, True],
            [True, False]
        ], dtype=bool)

        with self.assertRaises(Exception):
            map_receivers(
                mask,
                prevent_reciprocal=True
            )

    def test_single_participant_raises_exception(self):
        mask = np.array([
            [False]
        ], dtype=bool)

        with self.assertRaises(Exception):
            map_receivers(
                mask,
                prevent_reciprocal=True
            )
            
    
    def test_add_exclusion(self):
        hat = MagicHat()

        hat.add_exclusion(("Alice", "Bob"))

        # Names automatically added
        self.assertIn("Alice", hat.names)
        self.assertIn("Bob", hat.names)

        # Exclusion recorded
        self.assertIn(("Alice", "Bob"), hat.exclusions)

    def test_exclusion_is_directional(self):
        hat = MagicHat()

        hat.add_exclusion(("Alice", "Bob"))

        alice = hat.names.index("Alice")
        bob = hat.names.index("Bob")

        mask = hat.mask

        # Alice cannot give to Bob
        self.assertFalse(mask[alice, bob])

        # Bob can still give to Alice
        self.assertTrue(mask[bob, alice])

    def test_multiple_exclusions(self):
        hat = MagicHat()

        hat.add_exclusion(("Alice", "Bob"))
        hat.add_exclusion(("Charlie", "Alice"))

        mask = hat.mask

        alice = hat.names.index("Alice")
        bob = hat.names.index("Bob")
        charlie = hat.names.index("Charlie")

        self.assertFalse(mask[alice, bob])
        self.assertFalse(mask[charlie, alice])

        self.assertTrue(mask[bob, alice])
        self.assertTrue(mask[alice, charlie])

    def test_exclusion_inputs(self):
        hat = MagicHat()

        with self.assertRaises(TypeError):
            hat.add_exclusion(["Alice", "Bob"])

        with self.assertRaises(ValueError):
            hat.add_exclusion(("Alice",))

        with self.assertRaises(ValueError):
            hat.add_exclusion(("Alice", "Bob", "Charlie"))

        with self.assertRaises(TypeError):
            hat.add_exclusion(("Alice", 3))

    def test_multiple_exclusions(self):
        hat = MagicHat()

        pairings = [
            ("Alice", "Bob"),
            ("Charlie", "David"),
            ("Eve", "Alice")
        ]

        hat.add_exclusion(pairings)

        self.assertEqual(len(hat.exclusions), 3)

        mask = hat.mask

        alice = hat.names.index("Alice")
        bob = hat.names.index("Bob")

        # Directional restrictions
        self.assertFalse(mask[alice, bob])
        self.assertTrue(mask[bob, alice])

if __name__ == '__main__':
    unittest.main()
