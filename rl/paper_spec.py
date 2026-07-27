"""Single source of truth for what arXiv:2601.10825 actually specifies.

Every number here was read from the paper's HTML (v1) on 2026-07-26, not from notes.
`tests/test_paper_fidelity.py` asserts our runnable configs match this module, so a
drift between "what we run" and "what the paper says" fails a test instead of quietly
becoming a claim about replication.

Rule this file exists to enforce: if a value here is not matched by our config, we do
not call the run a replication. Deviations must be listed in DEVIATIONS with a reason.
"""
from __future__ import annotations

# --- Supplementary Table 9: the 8,262-problem pool the SFT priming data is drawn from ---
# NOTE: this is the pool for PRIMING. The RL task is Countdown, which is NOT in this pool.
# That out-of-domain relationship is the core of the paper's C5 claim.
POOL_BBH = {
    "boolean_expressions": 248, "causal_judgement": 173, "date_understanding": 195,
    "disambiguation_qa": 249, "formal_fallacies": 250, "geometric_shapes": 249,
    "hyperbaton": 247, "logical_deduction_five_objects": 244,
    "logical_deduction_seven_objects": 224, "logical_deduction_three_objects": 249,
    "movie_recommendation": 7, "navigate": 248, "object_counting": 247,
    "penguins_in_a_table": 143, "reasoning_about_colored_objects": 245,
    "ruin_names": 165, "salient_translation_error_detection": 249, "snarks": 146,
    "sports_understanding": 250, "temporal_sequences": 65,
    "tracking_shuffled_objects_five_objects": 112,
    "tracking_shuffled_objects_seven_objects": 102,
    "tracking_shuffled_objects_three_objects": 57, "web_of_lies": 72,
}
POOL_GPQA = {"gpqa_diamond": 161, "gpqa_extended": 474, "gpqa_main": 380}
POOL_MATH_HARD = {
    "algebra": 286, "counting_and_probability": 110, "geometry": 117,
    "intermediate_algebra": 212, "number_theory": 134, "prealgebra": 182,
    "precalculus": 104,
}
POOL_MUSR = {"murder_mysteries": 207, "object_placement": 256, "team_allocation": 247}
POOL_OTHER = {"ifeval": 524, "mmlu_pro": 432}

POOL_TOTAL = (sum(POOL_BBH.values()) + sum(POOL_GPQA.values())
              + sum(POOL_MATH_HARD.values()) + sum(POOL_MUSR.values())
              + sum(POOL_OTHER.values()))
assert POOL_TOTAL == 8262, f"pool must reconstruct the paper's 8,262; got {POOL_TOTAL}"

# --- the teacher that generates priming traces (Methods + Supplementary Table 7) ---
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# --- SFT priming data selection (main text) ---
SFT_N_TRAIN = 500
SFT_N_VAL = 100
SFT_SELECTION = "sample instances that reach correct answers"

# --- Supplementary Methods: SFT Data Generation Prompts (verbatim) ---
MONOLOGUE_PROMPT = (
    "{task}\n"
    "Answer: Enclose your step-by-step reasoning within <think> and </think> before "
    "answering. Do not answer directly without reasoning."
)
DIALOGUE_PROMPT = (
    "{task}\n"
    "Answer: You are simulating a collaborative group of thinkers solving a problem. "
    "Each thinker has a distinct persona and engages in a realistic, back-and-forth "
    "conversation. Thinkers may speak in any order and as many times as needed—no "
    "fixed turn order required. Keep all output strictly inside the tags defined "
    "below—no stray text. Present your answer between <group_solution> and "
    "</group_solution>. Do not try to be overly positive or polite during the "
    "conversation; focus on puzzle-solving, and note that disagreements can be helpful "
    "for the reasoning.\n"
    "Assume that there are {n_thinkers} thinkers and follow exactly the tag structure below:\n"
    "<cast_of_characters>\n"
    "{persona_slots}"
    "</cast_of_characters>\n"
    "<conversation>\n"
    "<!-- Each block is one utterance. Use <thinkX> … </thinkX> to indicate who is "
    "speaking. The order of speakers is entirely flexible. Thinkers can speak multiple "
    "times in a row. -->\n"
    "{think_slots}"
    "</conversation>\n"
    "<group_solution>\n"
    "Answer\n"
    "</group_solution>"
)
DIALOGUE_N_THINKERS = (2, 3, 4)  # "two, three, or four distinct personas"
DIALOGUE_ANSWER_TAG = "group_solution"
MONOLOGUE_ANSWER_TAG = "think"

# --- Supplementary Table 8: SFT hyperparameters ---
SFT_HPARAMS = {
    "train_val_size": (500, 100),
    "context_window": 2048,
    "train_batch_size": 64,
    "val_batch_size": 64,
    "optimizer": "adamw",
    "peak_lr": 1e-5,
    "warmup": "linear_10pct",
    "warmup_ratio": 0.10,
    "annealing": "cosine",
    "total_epochs": 5,
}

# --- Supplementary Table 6: PPO hyperparameters ---
PPO_HPARAMS = {
    "train_batch_size": 128,
    "val_batch_size": 640,
    "max_prompt_length": 1024,
    "max_response_length": 1024,
    "actor_lr": 1e-6,
    "critic_lr": 1e-5,
    "kl_coef": 0.001,
    "ppo_mini_batch_size": 64,
    "rollout_n": 4,
    "rollout_temperature": 1.0,
    "accuracy_weight": 0.9,
    "format_weight": 0.1,
    "total_steps": 250,
}

# --- RL task (Methods) ---
RL_TASK = "countdown"
RL_BASE_MODELS = ("Qwen/Qwen2.5-3B", "meta-llama/Llama-3.2-3B")
RL_ARMS = ("baseline", "dialogue", "monologue")
RL_VAL_N = 1024
RL_EVAL_EVERY = 10
RL_PROMPT = (
    "Using the numbers {numbers}, create an equation that equals {target}. You can use "
    "basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final answer in <answer> "
    "</answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)

# Llama-only: "reasoning content from multiple personas was concatenated into a single
# block (<think> </think>) to ensure comparable sequence lengths across conditions".
# Applies to Llama-3.2-3B ONLY -- the paper does NOT do this for Qwen.
LLAMA_CONCATENATE_PERSONAS = True

# --- Deviations we knowingly accept, with reasons. Anything not listed is a bug. ---
DEVIATIONS: dict[str, str] = {
    "pool_problem_identity": (
        "The paper gives per-subtask COUNTS (Table 9) that reconstruct to 8,262 exactly, "
        "but not WHICH problems it sampled where a benchmark is larger than its count "
        "(e.g. MMLU-Pro 432 of 12,032; GPQA main 380 of 448). We match the counts "
        "per subtask with a fixed seed. Sampling difference within an identical pool "
        "composition, not a design difference."
    ),
    "rl_prompt_scaffold": (
        "The paper quotes the Countdown INSTRUCTION only. We wrap it in TinyZero's "
        "conversation scaffold ending 'Assistant:' because verl's Countdown scorer "
        "locates the response by splitting on that marker and returns None (score 0) "
        "without it -- a bare instruction zeroes every rollout regardless of output. "
        "The paper's instruction text sits verbatim inside our prompt, so the scaffold "
        "adds only the marker. Corroborated: the paper reports a baseline PPO reward of "
        "0.5665 at 250 steps and our scaffolded baseline reached 0.597, while the bare "
        "prompt gives a structural 0.000."
    ),
    "gpqa_source_repo": (
        "The canonical GPQA repo (Idavidrein/gpqa) is gated and our HF token lacks "
        "access, so `rl.pool_build` falls back to the open mirror Wanfq/gpqa. The "
        "mirror carries identical per-config counts (diamond 198 / main 448 / extended "
        "546) and the same column schema, and the canonical repo is still tried first, "
        "so this changes the source of the bytes and not the problems. Remove this "
        "entry once the gate is accepted on the Hub."
    ),
}
