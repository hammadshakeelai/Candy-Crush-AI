"""
Comprehensive Exploratory Data Analysis (EDA) of Candy Crush Saga Levels.
Analyzes 10,299+ official King level blueprints, reverse-engineers game mechanics,
and generates publication-quality analytical charts in assets/eda/.
"""

import os
import json
from collections import Counter
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-DPI matplotlib configuration and modern aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e5e5e5'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

CONFIGS_PATH = "data/candycrush_repo/CCS_Scripts/json_extract/sample/configs_v1.208.json"
HEXAGON_PATH = "data/candycrush_repo/CCS_Scripts/hexagon/hl-v2-0.json"
ASSETS_DIR = "assets/eda"
DATA_EXPORT_PATH = "data/candycrush_eda_dataset.csv"

# Difficulty mapping
DIFFICULTY_MAP = {
    0: "Normal",
    1: "Hexagon (Hard)",
    2: "Super Hard",
    3: "Nightmare",
    4: "Legendary"
}

DIFFICULTY_PALETTE = {
    "Normal": "#4CAF50",         # Green
    "Hexagon (Hard)": "#2196F3",  # Blue
    "Super Hard": "#FF9800",     # Orange / Amber
    "Nightmare": "#9C27B0",      # Deep Purple / Magenta
    "Legendary": "#F44336"       # Crimson Red
}


def ep_lvl_to_total(e: int, l: int) -> int:
    """Computes continuous total level index from episode and level number."""
    if e <= 2:
        return 10 * (e - 1) + l
    else:
        return 20 + (e - 3) * 15 + l


def load_difficulty_ratings(hex_path: str) -> Dict[int, int]:
    """Loads King's official difficulty ratings (0-4) from hl-v2-0.json."""
    if not os.path.exists(hex_path):
        print(f"[Warning] Hexagon file {hex_path} not found. Proceeding without difficulty ratings.")
        return {}
    
    with open(hex_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    hl_list = data.get("hardlevels", [])
    diff_map = {}
    for entry in hl_list:
        e, l, d, u = entry.get("e", 0), entry.get("l", 0), entry.get("d", 0), entry.get("u", 1)
        lvl_num = ep_lvl_to_total(e, l)
        # Variation u=1 is standard production variation
        if lvl_num not in diff_map or u == 1:
            diff_map[lvl_num] = d
    print(f"[EDA] Loaded {len(diff_map)} difficulty ratings spanning level {min(diff_map.keys())} to {max(diff_map.keys())}.")
    return diff_map


def normalize_game_mode(raw_mode: str) -> str:
    """Normalizes King's raw gameModeName into clear canonical game modes."""
    if not raw_mode or not isinstance(raw_mode, str):
        return "Score"
    rm = raw_mode.lower()
    if "rainbow" in rm:
        return "Rainbow Rapids"
    if "jelly drop" in rm or "drop down jelly" in rm:
        return "Mixed (Jelly + Drop)"
    if "jelly order" in rm or "order jelly" in rm:
        return "Mixed (Jelly + Order)"
    if "order drop" in rm or "drop down order" in rm:
        return "Mixed (Order + Drop)"
    if "light up" in rm or "jelly" in rm:
        return "Jelly"
    if "drop" in rm:
        return "Drop down"
    if "order" in rm:
        return "Order"
    if "classic" in rm or "score" in rm:
        return "Score"
    return "Mixed"


def parse_tile_map(tile_map: List[List[str]]) -> Dict[str, Any]:
    """
    Parses a level's tileMap grid into spatial metrics and blocker counts.
    King tileMap encoding: each cell is a concatenated string of 3-digit tokens.
    """
    h = len(tile_map)
    w = len(tile_map[0]) if h > 0 else 0
    total_cells = h * w
    active_cells = 0
    void_cells = 0
    
    jelly_single = 0
    jelly_double = 0
    
    frosting_count = 0
    chocolate_regular = 0
    chocolate_dark = 0
    licorice_swirl = 0
    licorice_lock = 0
    toffee_swirl = 0
    cake_bomb_tiles = 0
    popcorn_tiles = 0
    skull_tiles = 0
    magic_mixer_tiles = 0
    frog_tiles = 0
    
    spatial_blocker_grid = np.zeros((h, w), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            cell_str = str(tile_map[r][c])
            tokens = [cell_str[i:i+3] for i in range(0, len(cell_str), 3)]
            
            if "000" in tokens or len(tokens) == 0:
                void_cells += 1
                continue
            
            active_cells += 1
            has_blocker_here = False
            
            # Jelly layers
            if "003" in tokens:
                jelly_single += 1
            if "004" in tokens:
                jelly_double += 1
                
            # Frosting (007, 021, 022, 023, 024)
            frost_tokens = {"007", "021", "022", "023", "024"}
            if any(t in frost_tokens for t in tokens):
                frosting_count += 1
                has_blocker_here = True
                
            # Chocolate regular (009, 010)
            if "009" in tokens or "010" in tokens:
                chocolate_regular += 1
                has_blocker_here = True
                
            # Dark chocolate (122, 123, 124, 125)
            dark_tokens = {"122", "123", "124", "125"}
            if any(t in dark_tokens for t in tokens):
                chocolate_dark += 1
                has_blocker_here = True
                
            # Licorice swirl (026)
            if "026" in tokens:
                licorice_swirl += 1
                has_blocker_here = True
                
            # Licorice lock (008, 017)
            if "008" in tokens or "017" in tokens:
                licorice_lock += 1
                has_blocker_here = True
                
            # Toffee swirl (080, 081, 082, 083)
            toffee_tokens = {"080", "081", "082", "083"}
            if any(t in toffee_tokens for t in tokens):
                toffee_swirl += 1
                has_blocker_here = True
                
            # Cake bomb (035)
            if "035" in tokens:
                cake_bomb_tiles += 1
                has_blocker_here = True
                
            # Popcorn (030, 109, 110, 111, 112)
            popcorn_tokens = {"030", "109", "110", "111", "112"}
            if any(t in popcorn_tokens for t in tokens):
                popcorn_tiles += 1
                has_blocker_here = True
                
            # Magic Mixer (062)
            if "062" in tokens:
                magic_mixer_tiles += 1
                has_blocker_here = True
                
            # Sour Skull (090)
            if "090" in tokens:
                skull_tiles += 1
                has_blocker_here = True
                
            # Candy Frog (078, 042)
            if "078" in tokens or "042" in tokens:
                frog_tiles += 1
                
            if has_blocker_here:
                spatial_blocker_grid[r, c] = 1.0

    total_jellies = jelly_single + jelly_double
    total_jelly_layers = jelly_single + (2 * jelly_double)
    total_blockers = (frosting_count + chocolate_regular + chocolate_dark + 
                      licorice_swirl + licorice_lock + toffee_swirl + 
                      cake_bomb_tiles + popcorn_tiles + skull_tiles + magic_mixer_tiles)
    
    blocker_density = total_blockers / max(1, active_cells)
    active_ratio = active_cells / max(1, total_cells)

    return {
        "height": h,
        "width": w,
        "total_cells": total_cells,
        "active_cells": active_cells,
        "void_cells": void_cells,
        "active_ratio": active_ratio,
        "jelly_single": jelly_single,
        "jelly_double": jelly_double,
        "total_jellies": total_jellies,
        "total_jelly_layers": total_jelly_layers,
        "frosting_count": frosting_count,
        "chocolate_regular": chocolate_regular,
        "chocolate_dark": chocolate_dark,
        "total_chocolate": chocolate_regular + chocolate_dark,
        "licorice_swirl": licorice_swirl,
        "licorice_lock": licorice_lock,
        "total_licorice": licorice_swirl + licorice_lock,
        "toffee_swirl": toffee_swirl,
        "cake_bomb_tiles": cake_bomb_tiles,
        "popcorn_tiles": popcorn_tiles,
        "skull_tiles": skull_tiles,
        "magic_mixer_tiles": magic_mixer_tiles,
        "frog_tiles": frog_tiles,
        "total_blockers": total_blockers,
        "blocker_density": blocker_density,
        "spatial_blocker_grid": spatial_blocker_grid
    }


def build_full_dataset(configs_path: str, hex_path: str) -> pd.DataFrame:
    """Loads all King configs and converts them into an analytical DataFrame."""
    print(f"[EDA] Reading {configs_path}...")
    with open(configs_path, "r", encoding="utf-8") as f:
        levels_raw = json.load(f)
    print(f"[EDA] Loaded {len(levels_raw)} level entries.")

    diff_map = load_difficulty_ratings(hex_path)
    records = []

    for idx, item in enumerate(levels_raw):
        gd = item.get("gameData", {})
        ep = item.get("episode", 0)
        lvl_in_ep = item.get("level", 0)
        
        # Determine continuous level number
        total_lvl = item.get("totalLevel")
        if total_lvl is None or total_lvl == 0:
            if ep > 0 and lvl_in_ep > 0:
                total_lvl = ep_lvl_to_total(ep, lvl_in_ep)
            else:
                total_lvl = idx + 1
        else:
            try:
                total_lvl = int(total_lvl)
            except (ValueError, TypeError):
                total_lvl = 10300 + idx

        raw_game_mode = gd.get("gameModeName", "Classic moves")
        norm_mode = normalize_game_mode(raw_game_mode)
        move_limit = gd.get("moveLimit", 30)
        num_colors = gd.get("numberOfColours", 5)

        # Portals
        portals = gd.get("portals", [])
        num_portals = len(portals) if isinstance(portals, list) else 0

        # Cannons & Ammo Cannons
        cannons = gd.get("cannons", [])
        num_cannons = sum(1 for c in cannons if isinstance(c, list) and len(c) > 0)
        ammo = gd.get("ammoCannons", [])
        num_ammo = sum(1 for a in ammo if isinstance(a, list) and len(a) > 0)

        # Evil Spawners (Magic Mixers)
        evil_elems = gd.get("evilSpawnerElements")
        has_spawner_config = bool(evil_elems and len(evil_elems) > 0)
        evil_interval = gd.get("evilSpawnerInterval", 3)
        evil_amount = gd.get("evilSpawnerAmount", 1)

        # Frog
        frog_stomach = gd.get("frogStomachSize", 0)
        has_frog = (frog_stomach is not None and frog_stomach > 0)

        # Target Scores
        scores = gd.get("scoreTargets", [10000, 20000, 30000])
        star1 = scores[0] if len(scores) > 0 else 10000
        star3 = scores[2] if len(scores) > 2 else (star1 * 3)

        # Tile Map Parsing
        tm = gd.get("tileMap", [])
        grid_metrics = parse_tile_map(tm)

        # Difficulty classification
        diff_id = diff_map.get(total_lvl, 0)
        diff_name = DIFFICULTY_MAP.get(diff_id, "Normal")

        record = {
            "level_number": total_lvl,
            "episode": ep,
            "level_in_episode": lvl_in_ep,
            "raw_game_mode": raw_game_mode,
            "game_mode": norm_mode,
            "move_limit": move_limit,
            "num_colors": num_colors,
            "difficulty_id": diff_id,
            "difficulty_name": diff_name,
            "has_portals": (num_portals > 0),
            "num_portals": num_portals,
            "has_cannons": (num_cannons > 0),
            "num_cannons": num_cannons,
            "has_ammo_cannons": (num_ammo > 0),
            "num_ammo_cannons": num_ammo,
            "has_magic_mixer": (has_spawner_config or grid_metrics["magic_mixer_tiles"] > 0),
            "magic_mixer_tiles": grid_metrics["magic_mixer_tiles"],
            "evil_interval": evil_interval if has_spawner_config else np.nan,
            "evil_amount": evil_amount if has_spawner_config else np.nan,
            "has_frog": (has_frog or grid_metrics["frog_tiles"] > 0),
            "frog_stomach": frog_stomach if has_frog else 0,
            "target_score_1star": star1,
            "target_score_3star": star3,
            **grid_metrics
        }
        records.append(record)

    df = pd.DataFrame(records)
    # Sort strictly by level_number
    df = df.sort_values("level_number").drop_duplicates(subset=["level_number"]).reset_index(drop=True)
    print(f"[EDA] Compiled cleaned DataFrame with {len(df)} levels and {len(df.columns)} features.")
    return df


def generate_eda_visualizations(df: pd.DataFrame, assets_dir: str):
    """Generates all 7 required presentation-grade figures."""
    os.makedirs(assets_dir, exist_ok=True)
    print(f"[EDA] Generating analytical figures in {assets_dir}...")

    # =========================================================================
    # 1. MOVE LIMITS DISTRIBUTION & HISTORICAL PROGRESSION
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: Histogram with KDE
    sns.histplot(
        data=df, x="move_limit", hue="difficulty_name",
        palette=DIFFICULTY_PALETTE, multiple="stack", bins=range(5, 65, 2),
        ax=axes[0], edgecolor="white", linewidth=0.5
    )
    axes[0].set_title("Distribution of Move Limits by Difficulty Tier", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("Move Limit Allotted", fontsize=12)
    axes[0].set_ylabel("Number of Levels", fontsize=12)
    axes[0].axvline(df['move_limit'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['move_limit'].mean():.1f} moves")
    axes[0].axvline(df['move_limit'].median(), color='black', linestyle=':', linewidth=1.5,
                   label=f"Median: {df['move_limit'].median():.0f} moves")
    axes[0].legend(title="Difficulty", loc="upper right")

    # Right: Level Progression Trendline (Rolling Mean)
    df_sorted = df.sort_values("level_number").copy()
    df_sorted['move_limit_rolling100'] = df_sorted['move_limit'].rolling(window=100, min_periods=10).mean()
    df_sorted['move_limit_rolling500'] = df_sorted['move_limit'].rolling(window=500, min_periods=50).mean()

    axes[1].scatter(df_sorted['level_number'], df_sorted['move_limit'],
                   color='#90caf9', alpha=0.15, s=6, label="Per-Level Move Limit")
    axes[1].plot(df_sorted['level_number'], df_sorted['move_limit_rolling100'],
                color='#1565c0', linewidth=1.8, label="100-Level Rolling Average")
    axes[1].plot(df_sorted['level_number'], df_sorted['move_limit_rolling500'],
                color='#d32f2f', linewidth=2.5, label="500-Level Macro Trend")
    axes[1].set_title("Evolution of Move Limits Across 10,000+ Level Progression", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Candy Crush Saga Level Number", fontsize=12)
    axes[1].set_ylabel("Move Limit", fontsize=12)
    axes[1].set_ylim(0, 65)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    chart1_path = os.path.join(assets_dir, "move_limits_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart1_path}")

    # =========================================================================
    # 2. GAME MODES DISTRIBUTION & PROGRESSION BREAKDOWN
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    mode_counts = df['game_mode'].value_counts()
    colors = sns.color_palette("Set2", len(mode_counts))

    # Donut Chart
    wedges, texts, autotexts = axes[0].pie(
        mode_counts, labels=mode_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors, wedgeprops=dict(width=0.45, edgecolor='w')
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_weight('bold')
    axes[0].set_title("Overall Game Modes Distribution\n(10,299+ Levels)", fontsize=14, fontweight='bold', pad=12)

    # 1000-Level Epoch Progression (Stacked Bar)
    df_sorted['epoch'] = (df_sorted['level_number'] // 1000) * 1000
    epoch_modes = pd.crosstab(df_sorted['epoch'], df_sorted['game_mode'], normalize='index') * 100

    epoch_modes.plot(
        kind='bar', stacked=True, colormap='tab10', ax=axes[1], edgecolor='white', linewidth=0.5
    )
    axes[1].set_title("Game Mode Share Progression by 1,000-Level Milestones", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Level Epoch (e.g. 1000 = Levels 1000-1999)", fontsize=12)
    axes[1].set_ylabel("Percentage of Levels (%)", fontsize=12)
    axes[1].set_xticklabels([f"L{int(i)}" for i in epoch_modes.index], rotation=45)
    axes[1].legend(title="Game Mode", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    chart2_path = os.path.join(assets_dir, "game_modes_distribution.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart2_path}")

    # =========================================================================
    # 3. BLOCKER DENSITY EVOLUTION OVER 10,000+ LEVELS
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)

    # Top: Total Blocker Density per Playable Cell
    df_sorted['blocker_density_roll'] = df_sorted['blocker_density'].rolling(150, min_periods=20).mean()
    axes[0].scatter(df_sorted['level_number'], df_sorted['blocker_density'],
                   color='#b0bec5', alpha=0.15, s=6, label="Per-Level Blocker Density")
    axes[0].plot(df_sorted['level_number'], df_sorted['blocker_density_roll'],
                color='#e65100', linewidth=2.5, label="150-Level Rolling Blocker Density")
    axes[0].set_title("Macro Blocker Density Evolution (Blocker Tiles / Active Playable Cells)", fontsize=14, fontweight='bold', pad=10)
    axes[0].set_ylabel("Blocker Density Ratio", fontsize=12)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left")

    # Bottom: Specific Complex Blocker Breakdown Across Levels
    blocker_cols = {
        "frosting_count": ("Frosting / Meringue", "#78909c"),
        "total_licorice": ("Licorice (Swirls & Locks)", "#212121"),
        "total_chocolate": ("Chocolate (Regular & Dark)", "#5d4037"),
        "toffee_swirl": ("Toffee Swirls", "#ffb300"),
        "magic_mixer_tiles": ("Magic Mixers", "#e91e63"),
        "cake_bomb_tiles": ("Cake Bombs (2x2)", "#ab47bc"),
        "popcorn_tiles": ("Popcorn / Kernel", "#ffd54f"),
    }

    for col, (label, color) in blocker_cols.items():
        rolling_series = df_sorted[col].rolling(200, min_periods=20).mean()
        axes[1].plot(df_sorted['level_number'], rolling_series, label=label, color=color, linewidth=2.0)

    axes[1].set_title("Emergence & Concentration of Complex Blockers (200-Level Smoothed Means)", fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel("Candy Crush Saga Level Number", fontsize=12)
    axes[1].set_ylabel("Average Tiles per Level", fontsize=12)
    axes[1].legend(loc="upper left", ncol=2)

    plt.tight_layout()
    chart3_path = os.path.join(assets_dir, "blocker_density_evolution.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart3_path}")

    # =========================================================================
    # 4. DIFFICULTY CLASSIFICATIONS & BUDGET ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Difficulty Breakdown Bar Chart
    diff_order = ["Normal", "Hexagon (Hard)", "Super Hard", "Nightmare", "Legendary"]
    diff_counts = df['difficulty_name'].value_counts().reindex(diff_order)
    sns.barplot(
        x=diff_counts.index, y=diff_counts.values,
        hue=diff_counts.index, legend=False,
        palette=[DIFFICULTY_PALETTE[d] for d in diff_order],
        ax=axes[0], edgecolor='white'
    )
    axes[0].set_title("Official King Difficulty Classifications", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_ylabel("Number of Levels", fontsize=12)
    axes[0].set_xticks(range(len(diff_order)))
    axes[0].set_xticklabels(diff_order, rotation=25)
    for i, v in enumerate(diff_counts.values):
        axes[0].text(i, v + 40, f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=9)

    # Move Limits Boxplot by Difficulty Tier
    sns.boxplot(
        data=df, x="difficulty_name", y="move_limit", order=diff_order,
        hue="difficulty_name", legend=False,
        palette=[DIFFICULTY_PALETTE[d] for d in diff_order], ax=axes[1],
        boxprops=dict(alpha=0.85), showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}
    )
    axes[1].set_title("Move Allowance by Difficulty Rating", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Difficulty Tier", fontsize=12)
    axes[1].set_ylabel("Moves Allowed", fontsize=12)
    axes[1].set_xticks(range(len(diff_order)))
    axes[1].set_xticklabels(diff_order, rotation=25)

    # Blocker Density Boxplot by Difficulty Tier
    sns.boxplot(
        data=df, x="difficulty_name", y="blocker_density", order=diff_order,
        hue="difficulty_name", legend=False,
        palette=[DIFFICULTY_PALETTE[d] for d in diff_order], ax=axes[2],
        boxprops=dict(alpha=0.85), showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}
    )
    axes[2].set_title("Blocker Density by Difficulty Rating", fontsize=14, fontweight='bold', pad=12)
    axes[2].set_xlabel("Difficulty Tier", fontsize=12)
    axes[2].set_ylabel("Blocker Density (Blockers / Active Cells)", fontsize=12)
    axes[2].set_xticks(range(len(diff_order)))
    axes[2].set_xticklabels(diff_order, rotation=25)

    plt.tight_layout()
    chart4_path = os.path.join(assets_dir, "difficulty_classifications.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart4_path}")

    # =========================================================================
    # 5. BOARD GEOMETRY & ACTIVE PLAYABLE CELL RATIOS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Grid Dimensions Heatmap (Height x Width)
    dim_matrix = pd.crosstab(df['height'], df['width'])
    sns.heatmap(dim_matrix, cmap="YlGnBu", annot=True, fmt='d', ax=axes[0], cbar_kws={'label': 'Level Count'})
    axes[0].set_title("Board Bounding Box Dimensions (Height x Width)", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("Board Width (Columns)", fontsize=12)
    axes[0].set_ylabel("Board Height (Rows)", fontsize=12)

    # Active Playable Cell Ratio Distribution
    sns.histplot(
        data=df, x="active_ratio", hue="difficulty_name",
        palette=DIFFICULTY_PALETTE, multiple="stack", bins=30,
        ax=axes[1], edgecolor='white'
    )
    axes[1].axvline(df['active_ratio'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean Active Ratio: {df['active_ratio'].mean():.2f}")
    axes[1].set_title("Active Playable Cell Ratios (Playable / Bounding Box)", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Active Ratio (1.0 = Solid Grid, <1.0 = Cutouts/Holes/Islands)", fontsize=12)
    axes[1].set_ylabel("Number of Levels", fontsize=12)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    chart5_path = os.path.join(assets_dir, "board_geometry_dimensions.png")
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart5_path}")

    # =========================================================================
    # 6. SPATIAL BLOCKER HEATMAP (9x9 BOARD PROBABILITY)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 8))

    # Aggregate spatial grids across standard 9x9 levels
    spatial_sum = np.zeros((9, 9), dtype=np.float32)
    valid_9x9_count = 0

    for _, row in df.iterrows():
        grid = row["spatial_blocker_grid"]
        if grid.shape == (9, 9):
            spatial_sum += grid
            valid_9x9_count += 1

    spatial_prob = (spatial_sum / max(1, valid_9x9_count)) * 100.0

    sns.heatmap(
        spatial_prob, annot=True, fmt=".1f", cmap="magma",
        ax=ax, cbar_kws={'label': 'Blocker Occurrence Probability (%)'},
        linewidths=1.0, linecolor='white'
    )
    ax.set_title(f"Spatial Blocker Placement Heatmap across {valid_9x9_count:,} Standard 9x9 Levels\n(Percentage of Levels with Blocker at (Row, Col))",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Board Column (0-8)", fontsize=11)
    ax.set_ylabel("Board Row (0-8, 0=Top, 8=Bottom)", fontsize=11)

    plt.tight_layout()
    chart6_path = os.path.join(assets_dir, "spatial_blocker_heatmap.png")
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart6_path}")

    # =========================================================================
    # 7. MACRO-MECHANICS FEATURE MATRIX & CORRELATIONS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Macro Mechanics Occurrence Counts
    mechanics_summary = {
        "Portals / Teleporters": df['has_portals'].sum(),
        "Cannons / Dispensers": df['has_cannons'].sum(),
        "Ammo Reservoirs": df['has_ammo_cannons'].sum(),
        "Magic Mixers (Spawners)": df['has_magic_mixer'].sum(),
        "Candy Frog": df['has_frog'].sum(),
        "Cake Bombs (2x2)": (df['cake_bomb_tiles'] > 0).sum(),
        "Popcorn Blockers": (df['popcorn_tiles'] > 0).sum(),
        "Sour Skulls": (df['skull_tiles'] > 0).sum()
    }
    mech_series = pd.Series(mechanics_summary).sort_values(ascending=True)

    axes[0].barh(mech_series.index, mech_series.values, color='#3f51b5', edgecolor='white')
    axes[0].set_title("Prevalence of Advanced Mechanics across 10,299+ Levels", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("Number of Levels Featuring Mechanic", fontsize=12)
    for i, v in enumerate(mech_series.values):
        axes[0].text(v + 50, i, f"{v:,} ({v/len(df)*100:.1f}%)", va='center', fontweight='bold', fontsize=10)

    # Feature Correlation Matrix
    corr_features = [
        "move_limit", "num_colors", "difficulty_id",
        "has_portals", "has_cannons", "has_magic_mixer",
        "total_jellies", "total_blockers", "blocker_density"
    ]
    corr_df = df[corr_features].rename(columns={
        "move_limit": "Moves", "num_colors": "Colors", "difficulty_id": "Difficulty",
        "has_portals": "Portals", "has_cannons": "Cannons", "has_magic_mixer": "Mixer",
        "total_jellies": "Jellies", "total_blockers": "Blockers", "blocker_density": "BlockerDens"
    }).corr()

    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1], cbar_kws={'label': 'Pearson Correlation'})
    axes[1].set_title("Correlation Matrix: Mechanics, Geometry, & Difficulty", fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    chart7_path = os.path.join(assets_dir, "mechanics_feature_matrix.png")
    plt.savefig(chart7_path, dpi=300)
    plt.close()
    print(f" -> Saved {chart7_path}")


def print_statistical_summary(df: pd.DataFrame):
    """Prints comprehensive statistics and tactical findings."""
    print("\n" + "="*80)
    print("      CANDY CRUSH SAGA 10,299+ LEVELS EDA STATISTICAL REPORT")
    print("="*80)
    print(f"Total Unique Levels Indexed: {len(df):,}")
    print(f"Level Range: Level {df['level_number'].min()} to Level {df['level_number'].max()}")
    print(f"Number of Episodes Covered: {df['episode'].max()}")
    
    print("\n--- DIFFICULTY TIER BREAKDOWN ---")
    for diff, cnt in df['difficulty_name'].value_counts().items():
        pct = cnt / len(df) * 100
        mean_moves = df[df['difficulty_name'] == diff]['move_limit'].mean()
        mean_block = df[df['difficulty_name'] == diff]['blocker_density'].mean()
        print(f"  {diff:<16}: {cnt:5,d} ({pct:5.1f}%) | Mean Moves: {mean_moves:4.1f} | Blocker Dens: {mean_block:.2f}")

    print("\n--- GAME MODE DISTRIBUTION ---")
    for mode, cnt in df['game_mode'].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"  {mode:<24}: {cnt:5,d} ({pct:5.1f}%)")

    print("\n--- MOVE LIMIT METRICS ---")
    print(f"  Min Moves   : {df['move_limit'].min()}")
    print(f"  Max Moves   : {df['move_limit'].max()}")
    print(f"  Mean Moves  : {df['move_limit'].mean():.2f}")
    print(f"  Median Moves: {df['move_limit'].median():.1f}")
    print(f"  Std Dev     : {df['move_limit'].std():.2f}")

    print("\n--- BOARD GEOMETRY & ACTIVE SPATIAL CELLS ---")
    std_grids = ((df['height'] == 9) & (df['width'] == 9)).sum()
    print(f"  Standard 9x9 Grids : {std_grids:,} ({std_grids / len(df) * 100:.1f}%)")
    print(f"  Mean Active Ratio  : {df['active_ratio'].mean():.3f} (Sparsity: {1 - df['active_ratio'].mean():.3f})")
    print(f"  Mean Playable Cells: {df['active_cells'].mean():.1f} / {df['total_cells'].mean():.1f}")

    print("\n--- ADVANCED MECHANICS FREQUENCY ---")
    print(f"  Portals / Teleporters   : {df['has_portals'].sum():5,d} ({df['has_portals'].mean()*100:4.1f}%)")
    print(f"  Cannons / Dispensers    : {df['has_cannons'].sum():5,d} ({df['has_cannons'].mean()*100:4.1f}%)")
    print(f"  Ammo Reservoirs         : {df['has_ammo_cannons'].sum():5,d} ({df['has_ammo_cannons'].mean()*100:4.1f}%)")
    print(f"  Magic Mixers (Spawners) : {df['has_magic_mixer'].sum():5,d} ({df['has_magic_mixer'].mean()*100:4.1f}%)")
    print(f"  Cake Bombs (2x2)        : {(df['cake_bomb_tiles']>0).sum():5,d} ({(df['cake_bomb_tiles']>0).mean()*100:4.1f}%)")
    print(f"  Toffee Swirls           : {(df['toffee_swirl']>0).sum():5,d} ({(df['toffee_swirl']>0).mean()*100:4.1f}%)")
    print(f"  Popcorn Machines        : {(df['popcorn_tiles']>0).sum():5,d} ({(df['popcorn_tiles']>0).mean()*100:4.1f}%)")
    print(f"  Candy Frog              : {df['has_frog'].sum():5,d} ({df['has_frog'].mean()*100:4.1f}%)")
    print("="*80 + "\n")


def main():
    df = build_full_dataset(CONFIGS_PATH, HEXAGON_PATH)
    
    # Save export CSV (without the heavy 2D numpy arrays)
    export_df = df.drop(columns=["spatial_blocker_grid"])
    os.makedirs(os.path.dirname(DATA_EXPORT_PATH), exist_ok=True)
    export_df.to_csv(DATA_EXPORT_PATH, index=False)
    print(f"[EDA] Exported tabular dataset to {DATA_EXPORT_PATH} ({os.path.getsize(DATA_EXPORT_PATH) / 1024 / 1024:.2f} MB)")
    
    print_statistical_summary(df)
    generate_eda_visualizations(df, ASSETS_DIR)
    print("[EDA] Analysis and visualizations completed successfully!")


if __name__ == "__main__":
    main()
