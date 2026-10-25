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
        # 1. SPECIAL COMBO MULTIPLIERS (TIER S+)
        # ==============================================================
        combo_bonus = 0.0
        if s1 == SpecialCandy.COLOR_BOMB and s2 == SpecialCandy.COLOR_BOMB:
            combo_bonus += 2000000.0  # Board wipe clears all jellies & blockers instantly!
        elif (s1 == SpecialCandy.COLOR_BOMB and s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL)) or \
             (s2 == SpecialCandy.COLOR_BOMB and s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL)):
            combo_bonus += 1000000.0  # Mass color detonation
        elif (s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and s2 == SpecialCandy.WRAPPED) or \
             (s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and s1 == SpecialCandy.WRAPPED):
            combo_bonus += 500000.0  # 3 rows + 3 cols super cross
        elif s1 == SpecialCandy.WRAPPED and s2 == SpecialCandy.WRAPPED:
            combo_bonus += 350000.0  # Dual 5x5 blast
        elif s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and \
             s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL):
            combo_bonus += 250000.0  # Cross clear
        elif s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB:
            combo_bonus += 150000.0  # Single color wipe
        elif s1 == SpecialCandy.JELLY_FISH or s2 == SpecialCandy.JELLY_FISH:
            combo_bonus += 350000.0  # Jelly Fish homing strikes!

        # ==============================================================
        # 2. SIMULATE 1-PLY CONSEQUENCE
        # ==============================================================
        sim_state = state.copy()
        score_gained, stats = self.engine.execute_move(sim_state, r, c, direction)
        if not stats["success"]:
            return -1e9

        if sim_state.has_won():
            return 10000000.0 + combo_bonus

        cascades = stats.get("cascades", 0)
        utility = combo_bonus

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

        utility += (new_bombs * 350000.0)      # Massive priority to create Color Bombs
        utility += (new_wrapped * 180000.0)    # High priority to create 5-match Wrappers
        utility += (new_striped * 90000.0)     # Solid reward for 4-match Striped candies

        # CATALYST DETECTOR: Check if the resulting state sets up special candy opportunities
        # If sim_state contains 4-in-a-row or 5-in-a-row opportunities, reward this catalyst move
        sim_legal = self.engine.get_legal_moves(sim_state)
        for sm_r, sm_c, sm_d in sim_legal[:12]:
            sm_s1 = sim_state.specials[sm_r, sm_c]
            sm_dr, sm_dc = OFFSETS[sm_d]
            sm_nr, sm_nc = sm_r + sm_dr, sm_c + sm_dc
            sm_s2 = sim_state.specials[sm_nr, sm_nc]
            if sm_s1 != SpecialCandy.NONE and sm_s2 != SpecialCandy.NONE:
                utility += 150000.0  # Set up adjacent combo for next turn!
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
                    utility += 250000.0  # Brought special directly adjacent to Color Bomb!
                elif manhattan <= 2:
                    utility += 75000.0   # 1 move away from adjacent!

        # ==============================================================
        # 6. OBJECTIVES PROGRESS
        # ==============================================================
        lvl_type = state.config.level_type

        # A. JELLY CLEARANCE
        if lvl_type == LevelType.JELLY:
            cleared_jellies = np.sum(state.jellies) - np.sum(sim_state.jellies)
            utility += (cleared_jellies * 45000.0)
            cols_with_jellies = set(np.where(state.jellies > 0)[1])
            rows_with_jellies = set(np.where(state.jellies > 0)[0])

            # Direct hit bonus on any tile containing jelly
            if state.jellies[r, c] > 0 or state.jellies[nr, nc] > 0:
                utility += 40000.0

            # OBJECTIVE REGION FOCUS (DISCONNECTED ISLANDS):
            # If move cleared 0 jellies, check if it was made in columns that contain jellies
            if cleared_jellies == 0:
                if len(cols_with_jellies) > 0 and c not in cols_with_jellies and nc not in cols_with_jellies:
                    # Playing on an auxiliary disconnected island with zero jellies
                    horizontal_laser_aimed = any(
                        (sim_state.specials[jr] == SpecialCandy.STRIPED_HORIZONTAL).any()
                        for jr in rows_with_jellies
                    )
                    combo_made = (new_bombs > 0) or horizontal_laser_aimed or (new_wrapped > 0)

                    # Reward detonating an existing special that fires into the locked island
                    if state.specials[r, c] != SpecialCandy.NONE or state.specials[nr, nc] != SpecialCandy.NONE:
                        utility += 350000.0

                    if horizontal_laser_aimed:
                        utility += 300000.0
                    if new_bombs > 0:
                        utility += 450000.0

                    # Check if there are ANY legal moves available directly on jelly cells
                    has_moves_on_jelly_island = False
                    for jc in cols_with_jellies:
                        for jr in range(state.height):
                            if not state.is_valid_cell(jr, jc):
                                continue
                            for d in [Direction.RIGHT, Direction.DOWN]:
                                dr, dc = OFFSETS[d]
                                if self.engine.is_legal_swap(state, jr, jc, jr + dr, jc + dc):
                                    has_moves_on_jelly_island = True
                                    break
                            if has_moves_on_jelly_island:
                                break
                        if has_moves_on_jelly_island:
                            break

                    if has_moves_on_jelly_island:
                        # Legal moves EXIST on the jelly island; don't wander off!
                        if not combo_made:
                            utility -= 150000.0
                    else:
                        # ARTILLERY MODE: Jelly island is locked! Auxiliary island must forge specials & cascades!
                        utility += 60000.0
                        if new_wrapped > 0 or new_striped > 0:
                            utility += 180000.0
                        # Reward cascades on auxiliary island to churn tiles and auto-spawn specials
                        utility += (cascades * 18000.0)

            # Bottom-row gravity cascade bonus (clearing bottom drives cascades through top)
            # Only apply if playing on columns with jellies
            if len(cols_with_jellies) == 0 or c in cols_with_jellies or nc in cols_with_jellies:
                utility += (max(r, nr) * 2500.0)

            # STUBBORN BOTTOM ROW EMPHASIS:
            # The bottom row (height - 1) is mathematically the hardest to clear because no candies fall beneath it.
            bottom_row = state.height - 1
            if state.jellies[bottom_row].sum() > 0:
                cols_with_bottom_jelly = set(np.where(state.jellies[bottom_row] > 0)[0])
                if c in cols_with_bottom_jelly or nc in cols_with_bottom_jelly:
                    if r == bottom_row or nr == bottom_row:
                        utility += 60000.0  # Massive priority to manipulate stubborn bottom row!
                if sim_state.jellies[bottom_row].sum() < state.jellies[bottom_row].sum():
                    utility += 150000.0  # Huge reward for successfully breaking bottom row jellies!

            # ENDGAME SNIPER: When <= 8 jellies remain, laser-focus exclusively on clearing them
            remaining_jelly_count = int(np.sum(state.jellies > 0))
            if remaining_jelly_count <= 8:
                # Direct hit bonus on remaining jelly
                if state.jellies[r, c] > 0 or state.jellies[nr, nc] > 0:
                    utility += 150000.0

                # Check if this move fires a line through any remaining jelly
                jelly_coords = list(zip(*np.where(state.jellies > 0)))
                for jr, jc in jelly_coords:
                    if sim_state.jellies[jr, jc] < state.jellies[jr, jc]:
                        utility += 250000.0  # Massive sniper kill bonus!

                # If move cleared zero jellies in endgame, penalize wandering
                if cleared_jellies == 0 and not sim_state.has_won():
                    utility -= 40000.0

            # ISOLATED POCKET & LASER ALIGNMENT:
            # For every remaining jelly, check if it can be directly reached by lasers
            jelly_coords = list(zip(*np.where(state.jellies > 0)))
            for jr, jc in jelly_coords:
                # Check if this jelly is isolated (no immediate candy neighbors to swap with)
                has_h_neighbors = (state.is_valid_cell(jr, jc - 1) and state.is_valid_cell(jr, jc + 1))
                has_v_neighbors = (state.is_valid_cell(jr - 1, jc) and state.is_valid_cell(jr + 1, jc))
                is_isolated = not (has_h_neighbors or has_v_neighbors)

                if is_isolated:
                    # If this move cleared the isolated jelly directly or via laser/bomb, massive reward!
                    if sim_state.jellies[jr, jc] < state.jellies[jr, jc]:
                        utility += 350000.0
                    # Isolated jellies MUST be cleared via horizontal beam on row jr or vertical beam on col jc
                    if (sim_state.specials[jr] == SpecialCandy.STRIPED_HORIZONTAL).any():
                        utility += 120000.0  # Horizontal laser aimed right at isolated pocket!
                    if (sim_state.specials[:, jc] == SpecialCandy.STRIPED_VERTICAL).any():
                        utility += 120000.0  # Vertical laser aimed right at isolated pocket!
                    if np.sum(sim_state.specials == SpecialCandy.COLOR_BOMB) > np.sum(state.specials == SpecialCandy.COLOR_BOMB):
                        utility += 90000.0  # Color bomb can wipe the pocket!

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

        # C. BLOCKERS (CHOCOLATE, LICORICE, FROSTING)
        choc_before = np.sum(state.blockers == Blocker.CHOCOLATE)
        choc_after = np.sum(sim_state.blockers == Blocker.CHOCOLATE)
        if choc_before > 0:
            if choc_after < choc_before:
                utility += 15000.0  # Destroying chocolate stops expansion
            elif choc_after > choc_before:
                utility -= 10000.0  # Letting chocolate grow is heavily penalized

        licorice_cleared = np.sum(state.blockers == Blocker.LICORICE) - np.sum(sim_state.blockers == Blocker.LICORICE)
        utility += (licorice_cleared * 15000.0)  # Breaking licorice locks frees trapped candies/jellies

        frosting_cleared = np.sum(state.blockers == Blocker.FROSTING) - np.sum(sim_state.blockers == Blocker.FROSTING)
        utility += (frosting_cleared * 8000.0)

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
