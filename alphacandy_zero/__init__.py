"""
AlphaCandy-Zero: search-centric expert-iteration agent for the ReverseCandyEngine.

Design principle (see knowledge_base/03_AI_Models_and_Paradigms/AlphaCandy_Zero_Design.md):
    We own an exact simulator, so the strongest achievable agent is
    "network-guided planning in the real simulator", trained by expert iteration
    (Gumbel AlphaZero, Danihelka et al. ICLR 2022), NOT a bigger model-free network.

Modules:
    encoding  - v2 state encoder (24 planes + 16 globals), action <-> index maps, flip symmetry
    network   - AlphaCandyZeroNet (S6 colour-invariant stem, FiLM-conditioned ResNet + axial
                attention, bilinear edge policy, win / final-objective / per-cell-jelly heads)
    search    - Gumbel sequential-halving root search over the true stochastic simulator
    train     - self-play expert iteration with Prioritized Level Replay + near-miss restarts
    evaluate  - honest evaluation protocol (per-level win rate over fixed seeds, no seed hunting)
    oracle    - clairvoyant determinized beam search: upper bound "is this level winnable at all
                in our engine?"  Separates engine/level-loading bugs from agent weakness.
"""
