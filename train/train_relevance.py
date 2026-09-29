"""Fine-tune Laya as a code-search relevance judge on one GPU.

    python train/train_relevance.py <cases.jsonl> <out_dir> [--epochs 3]

Adapted from Laya's own notebook (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
same RLCD loss (policy gradient with proper scoring rules + soft cross-entropy), same optimiser
settings, same held-out calibration fit. Differences: single GPU (no DDP), bf16 autocast (no
GradScaler), and the training items come from `build_dataset.py`, not typed-decisions.

Each case becomes two training items, one per question of the shipped `relevance` profile
(a yes/no "relevant" and a 0-3 "relevance" score), so training matches how the judge is called.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from layatools.profiles import SHIPPED_DIR, load_profile_file

BASE_MODEL = "convaiinnovations/laya"
EPOCHS_DEFAULT = 3
MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = 8, 8, 4  # effective batch 64, as in the notebook
LR_ENCODER, LR_HEAD = 2.5e-5, 1.0e-4
SIGMA_START, SIGMA_END = 0.4, 0.1
CALIB_FRACTION, CALIB_MAX = 0.1, 400


def build_items(cases, tok, cfg):
    profile = load_profile_file(SHIPPED_DIR / "relevance.yaml")
    items = []
    for case in cases:
        state = {"task": case["task"], "content": case["content"]}
        for spec in profile.questions.values():
            t, crit = spec["type"], spec.get("criteria", {})
            target = [1 - case["p_true"], case["p_true"]] if t == "noul" else list(case["levels"])
            k = len(render_options({"t": t, "crit": crit}))
            seq, markers = build_sequence(tok, state, {"t": t, "ins": spec["instructions"], "crit": crit},
                                          cfg["max_len"], cfg["head_max_len"])
            if len(markers) != k or len(target) != k:
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                          "label": target.index(max(target))})
    return items


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def forward(model, batch, device):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                            batch["qtype"].to(device))
    return logits.float(), act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    args = ap.parse_args()
    device = torch.device("cuda")

    model_dir = snapshot_download(BASE_MODEL)
    _fix_tokenizer_config(model_dir)
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
    cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096, max_len=1024, head_max_len=256)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    train_cases = [c for c in cases if c["split"] == "train"]
    all_items = build_items(train_cases, tok, cfg)
    print(f"{len(train_cases)} train cases -> {len(all_items)} items")

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    # Hold out a calibration slice before training: temperatures fitted on trained-on items measure
    # the fit, not the calibration.
    order = list(range(len(all_items)))
    random.Random(20260929).shuffle(order)
    n_calib = min(CALIB_MAX, int(len(all_items) * CALIB_FRACTION))
    calib = [all_items[i] for i in sorted(order[:n_calib])]
    train = [all_items[i] for i in sorted(order[n_calib:])]

    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    updates = max(1, (len(train) // (MICRO_BATCH * GRAD_ACCUM)) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)

    t0 = time.time()
    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(train)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, args.epochs - 1)
        opt.zero_grad(set_to_none=True)
        total, nb = 0.0, 0
        for b in range(0, len(train), MICRO_BATCH):
            batch = collate(train[b : b + MICRO_BATCH], tok.pad_token_id)
            logits, act = forward(model, batch, device)
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            loss.backward()
            nb += 1
            if nb % GRAD_ACCUM == 0 or b + MICRO_BATCH >= len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * GRAD_ACCUM
            if nb % 50 == 0:
                print(f"  epoch {epoch + 1}/{args.epochs} batch {nb} loss {loss.item() * GRAD_ACCUM:.4f} "
                      f"reward {r.mean().item():.3f} ({time.time() - t0:.0f}s)")
        print(f"=== epoch {epoch + 1} done, avg loss {total / max(1, nb):.4f} ({time.time() - t0:.0f}s) ===")

    print("fitting calibration temperatures on held-out items")
    model.eval()
    preds = []
    with torch.no_grad():
        for c in range(0, len(calib), 16):
            chunk = calib[c : c + 16]
            logits, _ = forward(model, collate(chunk, tok.pad_token_id), device)
            arr = logits.cpu().numpy()
            preds += [(it["qtype"], arr[i, : len(it["markers"])], it["target"]) for i, it in enumerate(chunk)]
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for qtype, z, t in preds if qtype == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out / "encoder"))
    tok.save_pretrained(str(out / "tokenizer"))
    cfg.update(fine_tuned=True, model_name="laya-code-relevance", temperature=temps)
    cfg.pop("temperature_by_options", None)  # inherited bucket overrides would hide the new fit
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
