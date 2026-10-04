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
PROMPT_2 = (
    "These are the first and last frames of a robot demonstration.\n"
    "Instruction given to the robot: \"{task}\"\n"
    "Looking at the last frame compared to the first, was the instruction fully completed? "
    "Answer with a single word: yes or no."
)
PROMPT_N = (
    "These are {n} frames in time order from one robot demonstration, from start to end.\n"
    "Instruction given to the robot: \"{task}\"\n"
    "By the final frame, has the instruction been fully completed? "
    "Answer with a single word: yes or no."
)


class Judge:
    def __init__(self, model_id: str = MODEL, device: str = "cuda", quant4: bool = False):
        self.processor = AutoProcessor.from_pretrained(model_id)
        if quant4:
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=q, device_map=device).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes"]})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No"]})

    @torch.inference_mode()
    def p_yes(self, first: Image.Image, last: Image.Image, task: str) -> float:
        return self.p_yes_frames([first, last], task)

    @torch.inference_mode()
    def p_yes_frames(self, frames: list[Image.Image], task: str) -> float:
        prompt = PROMPT_2 if len(frames) == 2 else PROMPT_N
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": f} for f in frames),
            {"type": "text", "text": prompt.format(task=task or "complete the task", n=len(frames))}]}]
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
    parser.add_argument("--n-frames", type=int, default=2, help="2 = first+last; N reads f0..f{N-1} from frames.py")
    parser.add_argument("--quant4", action="store_true", help="load the model in 4-bit (e.g. the 8B on 12 GB)")
    args = parser.parse_args()

    df = pd.read_parquet(args.scores, columns=["key", "source", "subset", "episode_index", "task"])
    if args.keys:
        df = df[df["key"].isin(pd.read_parquet(args.keys)["key"])]
    done = pd.read_parquet(args.out) if args.out.exists() else pd.DataFrame(columns=["key", "vlm_p_yes"])
    todo = df[~df["key"].isin(set(done["key"]))]
    judge = Judge(args.model, quant4=args.quant4)
    names = ["first", "last"] if args.n_frames == 2 else [f"f{k}" for k in range(args.n_frames)]
    rows = []
    for n, r in enumerate(todo.itertuples(), 1):
        d = args.frames / r.source / (r.subset or "_")
        paths = [d / f"{r.episode_index:06d}_{name}.jpg" for name in names]
        if not all(p.exists() for p in paths):
            continue
        rows.append(dict(key=r.key, vlm_p_yes=judge.p_yes_frames([Image.open(p) for p in paths], r.task)))
        if n % 200 == 0:  # checkpoint progress
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(args.out)
            rows = []
            print(f"{len(done)} judged", flush=True)
    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(args.out)


if __name__ == "__main__":
    main()
