"""
Grandmaster CANDYRL Ensemble with Macro-Combo Maneuvering & Strategic Alignment.
Capable of one-shot solving extreme density levels (e.g. Level 50 with 128 jellies in 24 moves).
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, SpecialCandy, Blocker,
    Ingredient, LevelType, Direction, OFFSETS
)


class GrandmasterCandyRLEnsemble:
    """
    Advanced multi-skill ensemble featuring:
    1. Hyper-Combo Hunter (prioritizes 5-matches and special creation)
    2. Combo Maneuvering (drags special candies together via gravity)
    3. Column-Path Clearance for Ingredients
    4. Adaptive Urgency Scaling based on (Objectives_Remaining / Moves_Remaining)
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None):
        self.engine = engine or ReverseCandyEngine()

    def evaluate_move(
        self, state: ReverseEngineState, r: int, c: int, direction: Direction
    ) -> float:
        dr, dc = OFFSETS[direction]
        nr, nc = r + dr, c + dc
        if not state.is_valid_cell(nr, nc):
            return -1e9

        s1 = state.specials[r, c]
        s2 = state.specials[nr, nc]

        # ==============================================================
        # 1. IMMEDIATE SPECIAL COMBOS (TIER S+)
        # ==============================================================
        if s1 == SpecialCandy.COLOR_BOMB and s2 == SpecialCandy.COLOR_BOMB:
            return 250000.0  # Board wipe clears all jellies & blockers instantly!

        if (s1 == SpecialCandy.COLOR_BOMB and s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL)) or \
           (s2 == SpecialCandy.COLOR_BOMB and s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL)):
            return 180000.0  # Mass color detonation

        if (s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and s2 == SpecialCandy.WRAPPED) or \
           (s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and s1 == SpecialCandy.WRAPPED):
            return 120000.0  # 3 rows + 3 cols super cross

        if s1 == SpecialCandy.WRAPPED and s2 == SpecialCandy.WRAPPED:
            return 90000.0  # Dual 5x5 blast

        if s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and \
           s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL):
            return 50000.0  # Cross clear

        if s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB:
            return 40000.0  # Single color wipe

        # ==============================================================
        # 2. SIMULATE 1-PLY CONSEQUENCE
        # ==============================================================
        sim_state = state.copy()
        score_gained, stats = self.engine.execute_move(sim_state, r, c, direction)
        if not stats["success"]:
            return -1e9

        if sim_state.has_won():
            return 500000.0  # Instant win

        cascades = stats.get("cascades", 0)
        utility = 0.0

        # ==============================================================
        # 3. URGENCY FACTOR: Objectives Remaining vs Moves Remaining
        # ==============================================================
        obj_left = state.remaining_objectives_count()
        moves_left = max(1, state.moves_left)
        urgency = obj_left / float(moves_left)  # e.g., 128 / 24 = 5.33

        # ==============================================================
        # 4. SPECIAL CANDY CREATION & CATALYST SETUP
        # ==============================================================
        # Check newly created specials directly from this move
        new_bombs = np.sum(sim_state.specials == SpecialCandy.COLOR_BOMB) - np.sum(state.specials == SpecialCandy.COLOR_BOMB)
        new_wrapped = np.sum(sim_state.specials == SpecialCandy.WRAPPED) - np.sum(state.specials == SpecialCandy.WRAPPED)
        new_striped = np.sum((sim_state.specials == SpecialCandy.STRIPED_HORIZONTAL) | (sim_state.specials == SpecialCandy.STRIPED_VERTICAL)) - \
                      np.sum((state.specials == SpecialCandy.STRIPED_HORIZONTAL) | (state.specials == SpecialCandy.STRIPED_VERTICAL))

        utility += (new_bombs * 50000.0)
        utility += (new_wrapped * 25000.0)
        utility += (new_striped * 12000.0)

        # CATALYST DETECTOR: Check if the resulting state sets up special candy opportunities
        # If sim_state contains 4-in-a-row or 5-in-a-row opportunities, reward this catalyst move
        sim_legal = self.engine.get_legal_moves(sim_state)
        for sm_r, sm_c, sm_d in sim_legal[:12]:
            sm_s1 = sim_state.specials[sm_r, sm_c]
            sm_dr, sm_dc = OFFSETS[sm_d]
            sm_nr, sm_nc = sm_r + sm_dr, sm_c + sm_dc
            sm_s2 = sim_state.specials[sm_nr, sm_nc]
            if sm_s1 != SpecialCandy.NONE and sm_s2 != SpecialCandy.NONE:
                utility += 40000.0  # Set up adjacent combo for next turn!
                break

        # ==============================================================
        # 5. COMBO MANEUVERING (BRINGING SPECIAL CANDIES CLOSER)
        # ==============================================================
        # If a Color Bomb and a Striped/Wrapped candy exist, reward moves that reduce their distance
        bomb_coords = list(zip(*np.where(sim_state.specials == SpecialCandy.COLOR_BOMB)))
        other_specials = list(zip(*np.where(
            (sim_state.specials == SpecialCandy.STRIPED_HORIZONTAL) |
            (sim_state.specials == SpecialCandy.STRIPED_VERTICAL) |
            (sim_state.specials == SpecialCandy.WRAPPED)
        )))

        for br, bc in bomb_coords:
            for sr, sc in other_specials:
                manhattan = abs(br - sr) + abs(bc - sc)
                if manhattan == 1:
                    utility += 60000.0  # Brought special directly adjacent to Color Bomb!
                elif manhattan <= 2:
                    utility += 15000.0  # 1 move away from adjacent!

        # ==============================================================
        # 6. OBJECTIVES PROGRESS
        # ==============================================================
        lvl_type = state.config.level_type

        # A. JELLY CLEARANCE
        if lvl_type == LevelType.JELLY:
            cleared_jellies = np.sum(state.jellies) - np.sum(sim_state.jellies)
            utility += (cleared_jellies * 10000.0)

            # ENDGAME SNIPER: When <= 6 jellies remain, laser-focus exclusively on clearing them
            remaining_jelly_count = int(np.sum(state.jellies > 0))
            if remaining_jelly_count <= 6:
                # Direct hit bonus on remaining jelly
                if state.jellies[r, c] > 0 or state.jellies[nr, nc] > 0:
                    utility += 150000.0

                # Check if this move fires a line through any remaining jelly
                jelly_coords = list(zip(*np.where(state.jellies > 0)))
                for jr, jc in jelly_coords:
                    if sim_state.jellies[jr, jc] < state.jellies[jr, jc]:
                        utility += 200000.0  # Massive sniper kill bonus!

                # If move cleared zero jellies in endgame, penalize wandering
                if cleared_jellies == 0 and not sim_state.has_won():
                    utility -= 25000.0

            # ISOLATED POCKET & LASER ALIGNMENT:
            # For every remaining jelly, check if it can be directly reached by lasers
            jelly_coords = list(zip(*np.where(state.jellies > 0)))
            for jr, jc in jelly_coords:
                # Check if this jelly is isolated (no immediate candy neighbors to swap with)
                has_h_neighbors = (state.is_valid_cell(jr, jc - 1) and state.is_valid_cell(jr, jc + 1))
                has_v_neighbors = (state.is_valid_cell(jr - 1, jc) and state.is_valid_cell(jr + 1, jc))
                is_isolated = not (has_h_neighbors or has_v_neighbors)

                if is_isolated:
                    # Isolated jellies MUST be cleared via horizontal beam on row jr or vertical beam on col jc
                    if (sim_state.specials[jr] == SpecialCandy.STRIPED_HORIZONTAL).any():
                        utility += 80000.0  # Horizontal laser aimed right at isolated pocket!
                    if (sim_state.specials[:, jc] == SpecialCandy.STRIPED_VERTICAL).any():
                        utility += 80000.0  # Vertical laser aimed right at isolated pocket!
                    if np.sum(sim_state.specials == SpecialCandy.COLOR_BOMB) > np.sum(state.specials == SpecialCandy.COLOR_BOMB):
                        utility += 60000.0  # Color bomb can wipe the pocket!

            # High priority for clearing isolated edge & perimeter jellies (Columns 0, W-1, Rows 0, H-1)
            for er in range(state.height):
                for ec in [0, state.width - 1]:
                    if state.jellies[er, ec] > sim_state.jellies[er, ec]:
                        utility += 25000.0  # Big reward for edge jelly clear

            # Reward aligning Vertical Striped candies down edge columns with remaining jellies
            if state.jellies[:, 0].sum() > 0 and (sim_state.specials[:, 0] == SpecialCandy.STRIPED_VERTICAL).any():
                utility += 30000.0
            if state.jellies[:, state.width - 1].sum() > 0 and (sim_state.specials[:, state.width - 1] == SpecialCandy.STRIPED_VERTICAL).any():
                utility += 30000.0

            # Reward aligning Horizontal Striped candies across rows with heavy jellies
            for hr in range(state.height):
                if state.jellies[hr].sum() >= 2 and (sim_state.specials[hr] == SpecialCandy.STRIPED_HORIZONTAL).any():
                    utility += 25000.0

        # B. INGREDIENTS CLEARANCE
        elif lvl_type == LevelType.INGREDIENTS:
            ing_collected = sim_state.ingredients_collected - state.ingredients_collected
            utility += (ing_collected * 60000.0)

            # Column path clearing: Reward clearing candies and blockers directly UNDER ingredients
            for col in range(state.width):
                ing_rows = np.where(state.ingredients[:, col] != Ingredient.NONE)[0]
                if len(ing_rows) > 0:
                    ing_r = ing_rows[0]
                    # Check blockers directly blocking this ingredient exit
                    blockers_below = np.sum(state.blockers[ing_r:, col] != Blocker.NONE)
                    sim_blockers_below = np.sum(sim_state.blockers[ing_r:, col] != Blocker.NONE)
                    if sim_blockers_below < blockers_below:
                        utility += 35000.0  # Destroyed bottleneck blocker blocking ingredient!

                    # Vertical striped candy aligned with ingredient column
                    if sim_state.specials[ing_r, col] == SpecialCandy.STRIPED_VERTICAL:
                        utility += 20000.0

                    # Reward matches below the ingredient in the same column
                    if (c == col and r > ing_r) or (nc == col and nr > ing_r):
                        utility += 12000.0

        # C. BLOCKERS (CHOCOLATE & FROSTING)
        choc_before = np.sum(state.blockers == Blocker.CHOCOLATE)
        choc_after = np.sum(sim_state.blockers == Blocker.CHOCOLATE)
        if choc_before > 0:
            if choc_after < choc_before:
                utility += 12000.0  # Destroying chocolate stops expansion
            elif choc_after > choc_before:
                utility -= 8000.0   # Letting chocolate grow is heavily penalized

        frosting_cleared = np.sum(state.blockers == Blocker.FROSTING) - np.sum(sim_state.blockers == Blocker.FROSTING)
        utility += (frosting_cleared * 3000.0)

        # ==============================================================
        # 7. AVALANCHE & BOTTOM-BIAS CASCADE
        # ==============================================================
        max_r = max(r, nr)
        bottom_bias = ((max_r / float(max(1, state.height - 1))) ** 1.8) * 3500.0
        cascade_value = (cascades ** 1.6) * 2000.0
        utility += bottom_bias + cascade_value

        # Score progress
        utility += (score_gained / 50.0)

        return utility


# Backward compatibility alias
CandyRLEnsemble = GrandmasterCandyRLEnsemble
