"""VLM success judge: does the final frame show the instruction completed?

Shows a local open VLM the first and last frames of an episode with its task
instruction and reads P(yes) from the next-token distribution over yes/no.
Validated against hand labels like the other detectors (curation/validate.py).

Usage:
    python -m curation.vlm_judge --scores AUDIT/scores.parquet --frames FRAMES_DIR --out AUDIT/vlm.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "These are the first and last frames of a robot demonstration.\n"
    "Instruction given to the robot: \"{task}\"\n"
    "Looking at the last frame compared to the first, was the instruction fully completed? "
    "Answer with a single word: yes or no."
)


class Judge:
    def __init__(self, model_id: str = MODEL, device: str = "cuda"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes"]})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No"]})

    @torch.inference_mode()
    def p_yes(self, first: Image.Image, last: Image.Image, task: str) -> float:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": first}, {"type": "image", "image": last},
            {"type": "text", "text": PROMPT.format(task=task or "complete the task")}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        y = torch.logsumexp(lp[self.yes_ids], 0)
        n = torch.logsumexp(lp[self.no_ids], 0)
        return float(torch.sigmoid(y - n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--keys", type=Path, help="optional parquet with a 'key' column to restrict to")
    args = parser.parse_args()

    df = pd.read_parquet(args.scores, columns=["key", "source", "subset", "episode_index", "task"])
    if args.keys:
        df = df[df["key"].isin(pd.read_parquet(args.keys)["key"])]
    done = pd.read_parquet(args.out) if args.out.exists() else pd.DataFrame(columns=["key", "vlm_p_yes"])
    todo = df[~df["key"].isin(set(done["key"]))]
    judge = Judge(args.model)
    rows = []
    for n, r in enumerate(todo.itertuples(), 1):
        d = args.frames / r.source / (r.subset or "_")
        first, last = d / f"{r.episode_index:06d}_first.jpg", d / f"{r.episode_index:06d}_last.jpg"
        if not (first.exists() and last.exists()):
            continue
        rows.append(dict(key=r.key, vlm_p_yes=judge.p_yes(Image.open(first), Image.open(last), r.task)))
        if n % 200 == 0:  # checkpoint progress
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(args.out)
            rows = []
            print(f"{len(done)} judged", flush=True)
    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(args.out)


if __name__ == "__main__":
    main()
