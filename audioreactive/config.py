"""Single source of truth for models, prompts, and tuning constants.

Every number that was previously scattered across generate.py / run_*.sh as a
magic literal lives here, so behavior changes happen in exactly one place and
the .json run metadata can quote this module instead of duplicating values.
"""

# Image models (both cached under ~/.cache/huggingface/hub, reused offline).
TURBO_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"  # 6B, 8 steps, fast checks
FULL_MODEL_ID = "Tongyi-MAI/Z-Image"  # 30 steps + guidance, final renders
MODEL_ID = TURBO_MODEL_ID  # CLI default; finals pass --model-id FULL_MODEL_ID

# Negative prompt tail every run script must include: blocks flat/monochrome
# blue skies that self-reinforce into solid-color runs over chained beats.
ANTI_BLUE_NEG = (
    "monochrome, monochromatic, solid blue, blue tint, cyan tint, flat blue"
)

# --- Beat chain tuning (see README "How it works") ---
KICK_GAIN = 0.30
SNARE_GAIN = 0.05
DRIFT = 0.03
ZMAX = 0.30
KICK_STEP = 0.05
BEAT_BLEND = 2
INTRO_FADE = 0.35
BEAT_STRENGTH = 0.4
BEATS_PER_PROMPT = 4

# --- Drift defenses (see AGENTS.md "Palette drift") ---
ANCHOR_PULL_L = 0.15
ANCHOR_PULL_S_LO = 0.40  # canvas duller than beat-0: pull up hard
ANCHOR_PULL_S_HI = 0.15  # canvas more vivid than beat-0: leave it alone
ANCHOR_PULL_AB = 0.10
TAME_BLUE_FRAC = 0.45
TAME_BLUE_DESAT = 0.70
PAN_MAX_STEP = 0.10
BEST_WINDOW_JITTER = 0.015

# Z-Image requires H/W divisible by 16.
DIM_MULTIPLE = 16
