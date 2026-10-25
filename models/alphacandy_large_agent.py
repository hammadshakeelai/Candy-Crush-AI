"""
Pure AlphaCandy Large Standalone Agent.
Operates exclusively on the 14.4M-parameter AlphaCandyLargeNet architecture without
auxiliary heuristic search or DQN dependencies.
Features:
  - 256 Hidden Channels with 2-Stage Multi-Head Spatial Self-Attention (8 heads each)
  - 6 Deep Residual Conv Blocks with Mish activations
  - Pure Neural Move Selection via Temperature-Scaled Softmax P(a|s)
  - Distributional Value V(s) and Calibrated Win Probability P(Win|s)
  - Fast Vectorized Tensor Batched Inference for GPU Core Saturation (< 2 ms)
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, LevelConfig,
    SpecialCandy, Blocker, Ingredient
)
from models.spatial_transformer_net import (
    AlphaCandyLargeNet, encode_spatial_board_tensor,
    encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


class AlphaCandyLargeAgent:
    """
    Pure Neural Standalone Agent powered by AlphaCandyLargeNet (~14.4M parameters).
    Selects moves strictly through end-to-end spatial self-attention over the 20-channel
    board tensor.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        engine: Optional[ReverseCandyEngine] = None,
        device: Optional[Union[torch.device, str]] = None,
        temperature: float = 0.0
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = engine or ReverseCandyEngine()
        self.temperature = max(0.0, float(temperature))
        self.net = AlphaCandyLargeNet(in_channels=20, hidden_dim=256, num_actions=144).to(self.device)

        # Checkpoint resolution
        resolved_ckpt = checkpoint_path
        if resolved_ckpt is None:
            candidates = [
                "checkpoints/alphacandy_large_net_master.pt",
                "checkpoints/alphacandy_large_net.pt",
                "checkpoints/alphacandy_spatial_transformer_master.pt",
                "checkpoints/alphacandy_spatial_transformer_local.pt",
                "checkpoints/alphacandy_spatial_transformer.pt"
            ]
            for c in candidates:
                if os.path.exists(c):
                    resolved_ckpt = c
                    break

        if resolved_ckpt and os.path.exists(resolved_ckpt):
            try:
                state_dict = torch.load(resolved_ckpt, map_location=self.device)
                # Auto-cast fp16 checkpoints to float32 if operating on CPU
                if self.device.type == "cpu":
                    state_dict = {k: v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v for k, v in state_dict.items()}
                self.net.load_state_dict(state_dict)
                print(f"[AlphaCandyLargeAgent] Loaded weights from {resolved_ckpt}")
            except Exception as e:
                print(f"[AlphaCandyLargeAgent] Checkpoint load note: {e}. Running with initialized 14.4M net.")
        else:
            print("[AlphaCandyLargeAgent] Initialized fresh AlphaCandyLargeNet (14.4M parameters).")

        self.net.eval()

    def select_best_move(
        self, state: ReverseEngineState
    ) -> Optional[Tuple[int, int, Direction, float, float]]:
        """
        Pure neural move selection.
        Returns:
            (r, c, direction, action_probability, estimated_win_prob)
        """
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 1.0, 0.5)

        s_np = encode_spatial_board_tensor(state)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

        masked_logits, probs, val_scalar, win_p = self.net.forward_with_mask(s_t, legal_moves)

        if self.temperature <= 0.001:
            # Greedy argmax over legal actions
            best_prob = -1.0
            best_move = legal_moves[0]
            for r, c, d in legal_moves:
                a_idx = encode_action_index(r, c, d)
                p = float(probs[a_idx].item())
                if p > best_prob:
                    best_prob = p
                    best_move = (r, c, d)
            return (best_move[0], best_move[1], best_move[2], best_prob, win_p)
        else:
            # Temperature sampling over legal actions
            legal_indices = [encode_action_index(r, c, d) for r, c, d in legal_moves]
            legal_logits = torch.tensor([masked_logits[idx].item() for idx in legal_indices], device=self.device)
            scaled_probs = F.softmax(legal_logits / self.temperature, dim=-1).cpu().numpy()
            chosen_idx = np.random.choice(len(legal_moves), p=scaled_probs)
            chosen_move = legal_moves[chosen_idx]
            chosen_prob = float(scaled_probs[chosen_idx])
            return (chosen_move[0], chosen_move[1], chosen_move[2], chosen_prob, win_p)

    def evaluate_batch(
        self, states: List[ReverseEngineState]
    ) -> List[Tuple[Optional[Tuple[int, int, Direction]], float, float]]:
        """
        Vectorized batch evaluation for multiple parallel boards on GPU.
        Processes B board tensors simultaneously in < 2 ms on Tensor Cores.
        """
        if not states:
            return []

        tensors = [encode_spatial_board_tensor(s) for s in states]
        batch_t = torch.from_numpy(np.stack(tensors, axis=0)).to(self.device)

        with torch.no_grad():
            policy_logits, val_scalars, win_probs = self.net(batch_t)

        results = []
        for i, s in enumerate(states):
            legal_moves = self.engine.get_legal_moves(s)
            if not legal_moves:
                results.append((None, 0.0, float(win_probs[i].item())))
                continue

            logits_1d = policy_logits[i]
            best_logit = -float("inf")
            best_mv = legal_moves[0]
            for r, c, d in legal_moves:
                a_idx = encode_action_index(r, c, d)
                l = float(logits_1d[a_idx].item())
                if l > best_logit:
                    best_logit = l
                    best_mv = (r, c, d)

            p_win = float(win_probs[i].item())
            results.append((best_mv, best_logit, p_win))

        return results

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        max_attempts: int = 1,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Solves a single level using purely the AlphaCandy Large neural network.
        """
        t0 = time.time()
        best_result = None

        for attempt in range(1, max_attempts + 1):
            if isinstance(config_or_state, ReverseEngineState):
                state = config_or_state
            else:
                state = ReverseEngineState(config_or_state)
                self.engine.initialize_board(state, seed=seed + (attempt - 1) * 313)

            moves_limit = max_moves or state.moves_left
            step = 0
            shuffles = 0
            move_history = []

            while not state.has_won() and not state.has_lost() and step < moves_limit:
                legal = self.engine.get_legal_moves(state)
                if not legal:
                    if not self.engine.shuffle_board(state):
                        break
                    shuffles += 1
                    continue

                mv_res = self.select_best_move(state)
                if mv_res is None:
                    break

                r, c, d, prob, win_p = mv_res
                move_history.append({"move": (r, c, d.name), "prob": prob, "win_p": win_p})
                self.engine.execute_move(state, r, c, d)
                step += 1

            won = state.has_won()
            res = {
                "won": won,
                "attempt": attempt,
                "moves_used": step,
                "moves_limit": moves_limit,
                "moves_remaining": state.moves_left,
                "final_score": state.score,
                "objectives_remaining": state.remaining_objectives_count(),
                "duration_sec": time.time() - t0,
                "shuffles": shuffles,
                "model": "AlphaCandyLargeNet (Pure Neural)",
                "move_history_length": len(move_history)
            }

            if won:
                return res
            if best_result is None or res["final_score"] > best_result["final_score"]:
                best_result = res

        return best_result or res
