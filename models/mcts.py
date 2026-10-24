"""
Information Set Monte Carlo Tree Search (ISMCTS) Solver for Candy Crush.
Handles stochastic candy drop transitions via determinized rollouts and UCT.
"""

from typing import List, Tuple, Dict, Optional, Any
import math
import random
import numpy as np
import torch
from core.candy_board import BoardState, Direction
from core.candy_engine import CandyEngine
from .heuristics import CandyCrushHeuristics


class MCTSNode:
    """A node in the MCTS tree representing a decision state."""

    def __init__(self, state: BoardState, parent: Optional["MCTSNode"] = None, move: Optional[Tuple[int, int, Direction]] = None):
        self.state = state
        self.parent = parent
        self.move = move  # The move that led to this node
        
        self.children: Dict[Tuple[int, int, Direction], "MCTSNode"] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior_prob: float = 1.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class CandyCrushMCTS:
    """
    Monte Carlo Tree Search with stochastic determinization for Candy Crush.
    Combines network priors or heuristic guidance with multi-step lookahead.
    """

    def __init__(
        self,
        engine: Optional[CandyEngine] = None,
        heuristics: Optional[CandyCrushHeuristics] = None,
        c_puct: float = 1.414,
        num_simulations: int = 50,
        rollout_depth: int = 3
    ):
        self.engine = engine if engine is not None else CandyEngine()
        self.heuristics = heuristics if heuristics is not None else CandyCrushHeuristics(self.engine)
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth

    def search(
        self,
        root_state: BoardState,
        policy_net: Optional[torch.nn.Module] = None
    ) -> Tuple[int, int, Direction]:
        """
        Runs MCTS simulations from the root state and returns the optimal move.
        """
        root = MCTSNode(root_state.copy())
        legal_moves = self.engine.get_legal_moves(root_state)
        
        if not legal_moves:
            # Fallback if no moves
            return (0, 0, Direction.RIGHT)
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Prior distribution initialization
        if policy_net is not None:
            with torch.no_grad():
                tensor = torch.from_numpy(root_state.to_tensor()).unsqueeze(0)
                logits, _ = policy_net(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        else:
            probs = None

        # Expand root
        for r, c, direction in legal_moves:
            child_state = root_state.copy()
            # Execute determinized step
            self.engine.execute_move(child_state, r, c, direction)
            child_node = MCTSNode(child_state, parent=root, move=(r, c, direction))
            
            if probs is not None:
                dir_idx = 0 if direction == Direction.RIGHT else 1
                action_idx = (r * root_state.width + c) * 2 + dir_idx
                child_node.prior_prob = float(probs[action_idx])
            else:
                child_node.prior_prob = 1.0 / len(legal_moves)
                
            root.children[(r, c, direction)] = child_node

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.is_expanded():
                node = self._select_child(node)
                search_path.append(node)

            # 2. Evaluation / Rollout
            value = self._simulate_rollout(node.state, self.rollout_depth)

            # 3. Backpropagation
            for visited_node in search_path:
                visited_node.visit_count += 1
                visited_node.total_value += value

        # Select child with highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Selects child using PUCT / UCB formula."""
        best_score = -float("inf")
        best_child = None

        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(max(1, total_visits))

        for child in node.children.values():
            q_val = child.value
            u_val = self.c_puct * child.prior_prob * sqrt_total / (1 + child.visit_count)
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child is not None else list(node.children.values())[0]

    def _simulate_rollout(self, state: BoardState, depth: int) -> float:
        """
        Fast heuristic rollout evaluating future board progression.
        """
        sim_state = state.copy()
        total_utility = 0.0

        for _ in range(depth):
            if sim_state.moves_left <= 0:
                break
            best = self.heuristics.select_best_move(sim_state)
            if best is None:
                break
            r, c, direction, score = best
            pts, jellies, stats = self.engine.execute_move(sim_state, r, c, direction)
            total_utility += (pts / 1000.0) + (jellies * 5.0)

        # Normalize utility
        return math.tanh(total_utility / 20.0)
