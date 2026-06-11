from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import block_size, tokenizer, device, vocab_size
import model


MODEL_PATH = Path(__file__).resolve().parent / "model_best.pth"


def load_model() -> model.GPTLanguageModel:
    m = model.GPTLanguageModel(device, vocab_size, block_size ).to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"missing checkpoint: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    m.load_state_dict(state_dict)
    m.eval()
    return m


@torch.no_grad()
def generate(
    m: model.GPTLanguageModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    idx = torch.tensor([tokenizer.encode(prompt, allowed_special="all")], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = m(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx], dim=1)
    return tokenizer.decode(idx[0].tolist())


def main() -> None:

    #torch.manual_seed(7)
    m = load_model()
    print(generate(m, "<|startoftext|> I'm an AI assistant and I", 200, 0.8))


if __name__ == "__main__":
    main()
