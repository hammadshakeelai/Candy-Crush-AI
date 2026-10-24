"""
Unit tests for Advanced Reverse-Engineered Mechanics and Master Level Database (23,355 levels).
"""

import unittest
import os
import sqlite3
import numpy as np
import torch

from core.advanced_mechanics import (
    PortalSystem, CannonDispenserSystem, MagicMixerState,
    CakeBombState, CandyFrogState, PopcornState, ToffeeSwirlState,
    AdvancedMechanicsManager, AdvancedBlockerType
)
from core.king_level_loader import KingLevelLoader
from models.network import CandyCrushPolicyValueNet, encode_advanced_spatial_tensor


class TestAdvancedMechanics(unittest.TestCase):

    def test_master_level_database_exists_and_complete(self):
        """Verify that data/candycrush_master_levels.sqlite contains all 23,355 levels."""
        db_path = "data/candycrush_master_levels.sqlite"
        self.assertTrue(os.path.exists(db_path), f"Database not found at {db_path}")

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT count(*) FROM levels")
        total = c.fetchone()[0]
        self.assertEqual(total, 23355, f"Expected 23,355 levels, found {total}")

        # Check sample levels
        c.execute("SELECT level_id, game_mode, move_limit, difficulty_label FROM levels WHERE level_id = 1")
        row = c.fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "Order")
        self.assertEqual(row[2], 28)

        c.execute("SELECT level_id, game_mode, move_limit FROM levels WHERE level_id = 23045")
        row = c.fetchone()
        self.assertEqual(row[0], 23045)
        self.assertEqual(row[2], 20)

        conn.close()

    def test_portal_system(self):
        """Test portal routing between disjoint board locations."""
        # Portals format: [ [[src_x, src_y, 14], [dst_x, dst_y, 14]] ]
        raw_portals = [[[2, 8, 14], [2, 0, 14]], [[5, 8, 14], [5, 0, 14]]]
        ps = PortalSystem(raw_portals)

        self.assertTrue(ps.has_portal(8, 2))
        self.assertEqual(ps.get_destination(8, 2), (0, 2))
        self.assertEqual(ps.get_source(0, 2), (8, 2))
        self.assertFalse(ps.has_portal(4, 4))

    def test_cannon_dispenser_system(self):
        """Test cannons and ammunition tracking."""
        raw_cannons = [[26], [], [71]]  # col 0 has 26, col 2 has 71
        raw_ammo = [5, 0, 3]
        cds = CannonDispenserSystem(raw_cannons, raw_ammo)

        self.assertTrue(cds.has_dispenser(0))
        self.assertFalse(cds.has_dispenser(1))
        self.assertTrue(cds.has_dispenser(2))
        self.assertEqual(cds.ammo_counts[0], 5)

    def test_magic_mixer_state(self):
        """Test Magic Mixer hit damage, stun state, and spew cooldown."""
        mixer = MagicMixerState(row=4, col=4, hp=4, interval=3, countdown=3)

        # Initial state
        self.assertEqual(mixer.hp, 4)
        self.assertEqual(mixer.countdown, 3)

        # Deal hit damage
        destroyed = mixer.take_hit()
        self.assertFalse(destroyed)
        self.assertEqual(mixer.hp, 3)
        self.assertTrue(mixer.was_damaged_this_turn)
        self.assertEqual(mixer.countdown, 3)  # Reset to interval

        # Step turn while stunned does not countdown
        ready = mixer.step_turn()
        self.assertFalse(ready)
        self.assertFalse(mixer.was_damaged_this_turn)

        # Step turn without damage decrements countdown
        ready = mixer.step_turn()
        self.assertFalse(ready)
        self.assertEqual(mixer.countdown, 2)

    def test_cake_bomb_state(self):
        """Test Cake Bomb 2x2 footprint and 8-slice removal."""
        cake = CakeBombState(top_left_r=3, top_left_c=3, slices_remaining=8)

        self.assertTrue(cake.is_cell_part_of_cake(3, 3))
        self.assertTrue(cake.is_cell_part_of_cake(4, 4))
        self.assertFalse(cake.is_cell_part_of_cake(2, 2))

        # Hit 7 slices
        for _ in range(7):
            wiped = cake.take_hit(1)
            self.assertFalse(wiped)

        # 8th hit triggers explosion
        wiped = cake.take_hit(1)
        self.assertTrue(wiped)
        self.assertEqual(cake.slices_remaining, 0)

    def test_candy_frog_state(self):
        """Test Candy Frog feeding and Frogtastic activation."""
        frog = CandyFrogState(row=0, col=0, color=2, stomach_capacity=6)

        self.assertFalse(frog.is_frogtastic)

        # Feed 6 candies
        is_ready = frog.feed(count=6)
        self.assertTrue(is_ready)
        self.assertTrue(frog.is_frogtastic)

        # Frogtastic leap to (4, 4)
        pos = frog.jump_and_crush(target_r=4, target_c=4, new_color=3)
        self.assertEqual(pos, (4, 4))
        self.assertEqual(frog.color, 3)
        self.assertFalse(frog.is_frogtastic)
        self.assertEqual(frog.current_fullness, 0)

    def test_resnet_28_channel_tensor_and_action_masking(self):
        """Verify 28-channel ResNet tensor encoding and action masking forward pass."""
        net = CandyCrushPolicyValueNet(in_channels=28, num_res_blocks=2, channels=32)

        x = torch.randn(2, 28, 9, 9)
        policy_logits, value = net(x)

        self.assertEqual(policy_logits.shape, (2, 162))
        self.assertEqual(value.shape, (2, 1))

        # Action masking
        masks = torch.ones(2, 162, dtype=torch.bool)
        masks[:, 10:] = False  # Only first 10 actions valid

        masked_logits, probs, val = net.forward_with_mask(x, masks)
        self.assertEqual(masked_logits.shape, (2, 162))
        self.assertEqual(val.shape, (2, 1))
        # Masked actions must have 0 probability
        self.assertTrue(torch.allclose(probs[:, 10:], torch.zeros_like(probs[:, 10:])))
        self.assertAlmostEqual(probs[0].sum().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
