"""SFT priming must train on the RESPONSE, never the prompt.

The single silent corruptor of instruction SFT is loss masking. If the prompt
tokens are not masked out, the model is trained to also generate the Countdown
prompt -- diluting the dialogue-vs-monologue signal that Claim B depends on, and
doing it invisibly (loss still goes down, checkpoints still save). So the masking
is the one thing worth pinning before any GPU is touched.

These tests are pure list arithmetic plus a stub tokenizer -- no torch, no model.
"""

from __future__ import annotations

from rl.sft_prime import encode_example, mask_prompt_labels

IGN = -100


def test_prompt_tokens_are_masked_response_tokens_are_kept():
    ids = [10, 11, 12, 20, 21]  # first 3 = prompt, last 2 = response
    labels = mask_prompt_labels(ids, prompt_len=3)
    assert labels == [IGN, IGN, IGN, 20, 21]


def test_nothing_is_masked_when_prompt_is_empty():
    assert mask_prompt_labels([5, 6], prompt_len=0) == [5, 6]


def test_everything_is_masked_when_response_is_empty():
    """A fully-masked example contributes no loss; better that than training on the
    prompt. Guards the truncation edge where the response fell off max_length."""
    assert mask_prompt_labels([5, 6], prompt_len=2) == [IGN, IGN]


def test_mask_never_exceeds_sequence_length():
    assert mask_prompt_labels([5, 6], prompt_len=9) == [IGN, IGN]


class _StubTok:
    """Deterministic word-id tokenizer: each whitespace token -> a stable int id.
    eos_token_id appended by encode_example, mirroring a real causal-LM SFT."""
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + (len(w) % 50) for w in text.split()]}


def test_encode_masks_exactly_the_prompt_span_and_appends_eos():
    tok = _StubTok()
    prompt, response = "the shared prompt", "dialogue answer here"
    ex = encode_example(tok, prompt, response, max_len=64)

    n_prompt = len(tok(prompt)["input_ids"])
    # response tokens + appended eos are supervised; prompt tokens are not
    assert ex["labels"][:n_prompt] == [IGN] * n_prompt
    assert all(l != IGN for l in ex["labels"][n_prompt:])
    assert ex["input_ids"][-1] == tok.eos_token_id
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    assert ex["labels"][-1] == tok.eos_token_id  # the model must learn to stop


def test_encode_truncates_to_max_len_without_desyncing_labels():
    tok = _StubTok()
    ex = encode_example(tok, "a b c d e", "f g h i j k", max_len=4)
    assert len(ex["input_ids"]) == 4
    assert len(ex["labels"]) == 4
    assert len(ex["attention_mask"]) == 4
