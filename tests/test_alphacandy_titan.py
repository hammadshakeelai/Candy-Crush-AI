"""
Unit tests for AlphaCandyTitanNet and its architectural submodules.
"""

import unittest
import torch
from models.alphacandy_titan import (
    AlphaCandyTitanNet,
    ColorSymmetricRelationalStem,
    BilinearEdgeSwapHead,
    DynamicQuantileValueHead,
    TelemetryConditioner,
)


class TestAlphaCandyTitan(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.dummy_board = torch.randn(self.batch_size, 20, 9, 9, device=self.device)

    def test_titan_compact_forward_pass(self):
        """Verify full forward pass and output shapes of AlphaCandyTitanNet with compact dimensions."""
        model = AlphaCandyTitanNet(
            in_channels=20,
            board_h=9,
            board_w=9,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_mlp=128,
            k_ponder=2,
            num_quantiles=32,
            num_actions=144,
        ).to(self.device)
        model.eval()

        with torch.no_grad():
            out = model(self.dummy_board, moves_left=15, objectives_left=8)

        self.assertIn("policy_logits", out)
        self.assertIn("q_quantiles", out)
        self.assertIn("q_expected", out)
        self.assertIn("win_prob", out)
        self.assertIn("hazard_map", out)
        self.assertIn("epistemic_confidence", out)

        self.assertEqual(out["policy_logits"].shape, (self.batch_size, 144))
        self.assertEqual(out["q_quantiles"].shape, (self.batch_size, 144, 32))
        self.assertEqual(out["q_expected"].shape, (self.batch_size, 144))
        self.assertEqual(out["win_prob"].shape, (self.batch_size, 1))
        self.assertEqual(out["hazard_map"].shape, (self.batch_size, 4, 9, 9))

        conf = out["epistemic_confidence"]
        self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_color_symmetric_relational_stem(self):
        """Verify ColorSymmetricRelationalStem produces expected (B, d_model, H, W) projection."""
        stem = ColorSymmetricRelationalStem(
            num_colors=6,
            context_channels=14,
            color_feat_dim=32,
            d_model=64,
        )
        out = stem(self.dummy_board)
        self.assertEqual(out.shape, (self.batch_size, 64, 9, 9))

    def test_telemetry_conditioner(self):
        """Verify TelemetryConditioner outputs modulation vector and telemetry features."""
        conditioner = TelemetryConditioner(telemetry_dim=16, d_model=64)
        c_telem, telem_vec, phase_probs = conditioner(
            self.dummy_board, moves_left=20, objectives_left=10
        )
        self.assertEqual(c_telem.shape, (self.batch_size, 64))
        self.assertEqual(telem_vec.shape, (self.batch_size, 16))
        self.assertEqual(phase_probs.shape, (self.batch_size, 3))

    def test_dynamic_quantile_head_risk_modes(self):
        """Verify DynamicQuantileValueHead risk tilting across safe, normal, and miracle regimes."""
        d_model = 64
        edge_hidden_dim = 64
        quantile_head = DynamicQuantileValueHead(
            d_model=d_model, edge_hidden_dim=edge_hidden_dim, num_quantiles=32, num_actions=144
        )
        z = torch.randn(self.batch_size, 81, d_model)
        edge_embeds = torch.randn(self.batch_size, 144, edge_hidden_dim)
        c_telemetry = torch.randn(self.batch_size, d_model)

        q_quantiles, q_expected = quantile_head(z, edge_embeds, c_telemetry)
        self.assertEqual(q_quantiles.shape, (self.batch_size, 144, 32))
        self.assertEqual(q_expected.shape, (self.batch_size, 144))

        # Safe mode
        score_safe, mode_safe, _ = quantile_head.compute_pace_tilted_scores(
            q_quantiles, moves_left=30, objectives_left=2
        )
        self.assertEqual(score_safe.shape, (self.batch_size, 144))
        self.assertEqual(mode_safe, "SAFE_CVAR20")

        # Normal mode
        score_norm, mode_norm, _ = quantile_head.compute_pace_tilted_scores(
            q_quantiles, moves_left=20, objectives_left=15
        )
        self.assertEqual(score_norm.shape, (self.batch_size, 144))
        self.assertEqual(mode_norm, "NORMAL_EV")

        # Miracle mode
        score_miracle, mode_miracle, _ = quantile_head.compute_pace_tilted_scores(
            q_quantiles, moves_left=1, objectives_left=20
        )
        self.assertEqual(score_miracle.shape, (self.batch_size, 144))
        self.assertEqual(mode_miracle, "MIRACLE_UPPER_CVAR85")


if __name__ == "__main__":
    unittest.main()
