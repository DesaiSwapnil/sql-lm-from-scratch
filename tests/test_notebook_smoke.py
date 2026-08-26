"""
Smoke test for notebooks/train_sql_lm.ipynb.

Runs every code cell's logic sequentially (Colab-specific lines omitted:
Drive mount, pip install, git clone). Verifies each stage either skips
correctly or runs and produces its expected output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)
os.environ["PYTHONPATH"] = REPO


# ── Cell: Configuration ───────────────────────────────────────────────────────

SMOKE = True

DATA_DIR        = "/tmp/sql_lm_smoke"
CKPT_DIR        = "/tmp/sql_lm_smoke"
LOG_DIR         = "/tmp/sql_lm_smoke/logs"
HF_CACHE        = "/tmp/hf_cache"
DRIVE_CKPT      = ""
DRIVE_DATA      = ""
SPIDER_DIR      = ""
CONFIG_BASE     = "configs/smoke/base.json"
CONFIG_PRETRAIN = "configs/smoke/pretrain.json"
CONFIG_SFT      = "configs/smoke/sft.json"
CONFIG_GRPO     = "configs/smoke/grpo.json"

for d in [DATA_DIR, CKPT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

PRETRAIN_CKPT = os.path.join(CKPT_DIR, "pretrain.pt")
SFT_CKPT      = os.path.join(CKPT_DIR, "sft.pt")
GRPO_CKPT     = os.path.join(CKPT_DIR, "grpo.pt")

def _cfg_int(path, key, default=0):
    try:
        return json.loads(open(path).read()).get(key, default) or default
    except Exception:
        return default

PRETRAIN_TOTAL = _cfg_int(CONFIG_PRETRAIN, "train_steps")
GRPO_TOTAL     = _cfg_int(CONFIG_GRPO,     "iterations")
SFT_EPOCHS     = _cfg_int(CONFIG_SFT,      "epochs")

print(f"[config] SMOKE={SMOKE}  pretrain_total={PRETRAIN_TOTAL}  "
      f"sft_epochs={SFT_EPOCHS}  grpo_total={GRPO_TOTAL}")


# ── Cell: Helpers ─────────────────────────────────────────────────────────────

def _run(cmd: list) -> None:
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, "PYTHONPATH": REPO},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command exited {proc.returncode}: {' '.join(str(c) for c in cmd)}")


def _ckpt_is_valid(path: str, expected_stage: str = None) -> bool:
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "stage" not in ckpt or "step" not in ckpt:
            return False
        if expected_stage and ckpt.get("stage") != expected_stage:
            return False
        return True
    except Exception:
        return False


def _training_needed(path: str, stage: str, total_steps: int = None,
                     sft_mode: bool = False):
    if not os.path.exists(path):
        return True, False
    if not _ckpt_is_valid(path, stage):
        print(f"  [warn] Corrupt checkpoint at {path!r} — removing and restarting.")
        os.remove(path)
        return True, False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if sft_mode:
        done_epochs = ckpt.get("epoch", 0)
        cfg_epochs  = ckpt.get("cfg", {}).get("epochs", SFT_EPOCHS)
        if done_epochs >= cfg_epochs:
            print(f"  SFT complete ({done_epochs}/{cfg_epochs} epochs) — skipping.")
            return False, False
        print(f"  SFT epoch {done_epochs}/{cfg_epochs} — will resume.")
        return True, True
    if total_steps is not None:
        done = ckpt.get("step", 0)
        if done >= total_steps:
            print(f"  Complete (step {done}/{total_steps}) — skipping.")
            return False, False
        print(f"  Step {done}/{total_steps} — will resume.")
        return True, True
    return True, True


def _data_is_ready(local_path: str, drive_dir: str = "") -> bool:
    marker_local = local_path + ".done"
    if os.path.exists(marker_local) and os.path.exists(local_path):
        return True
    if drive_dir:
        marker_drive = os.path.join(drive_dir, os.path.basename(marker_local))
        drive_copy   = os.path.join(drive_dir, os.path.basename(local_path))
        if os.path.exists(marker_drive) and os.path.exists(drive_copy):
            print(f"  [cache] Restoring {os.path.basename(local_path)} from Drive …")
            shutil.copy2(drive_copy,   local_path)
            shutil.copy2(marker_drive, marker_local)
            return True
    return False


def _mark_done(local_path: str, drive_dir: str = "") -> None:
    marker = local_path + ".done"
    open(marker, "w").close()
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy2(local_path, os.path.join(drive_dir, os.path.basename(local_path)))
        shutil.copy2(marker,     os.path.join(drive_dir, os.path.basename(marker)))
        print(f"  [sync] Drive ← {os.path.basename(local_path)}")


print("[helpers] defined")


# ── Tracking ──────────────────────────────────────────────────────────────────

results: list[tuple[str, str, float]] = []   # (cell, status, elapsed_s)

def _section(name: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

def _ok(cell: str, elapsed: float, note: str = "") -> None:
    tag = f"  [{note}]" if note else ""
    print(f"  ✓ {cell}{tag}  ({elapsed:.1f}s)")
    results.append((cell, "ok", elapsed))

def _fail(cell: str, elapsed: float, err: str) -> None:
    print(f"  ✗ {cell}  ({elapsed:.1f}s): {err}")
    results.append((cell, "FAIL", elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4a: Pretrain data
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 4a: Pretrain data")
t0 = time.perf_counter()

PRETRAIN_TRAIN = os.path.join(DATA_DIR, "pretrain_train.h5")
PRETRAIN_DEV   = os.path.join(DATA_DIR, "pretrain_dev.h5")

try:
    if _data_is_ready(PRETRAIN_TRAIN, DRIVE_DATA) and _data_is_ready(PRETRAIN_DEV, DRIVE_DATA):
        _ok("pretrain-data", time.perf_counter() - t0, "skipped — .done present")
    else:
        cmd = [sys.executable, "scripts/prepare_pretrain_data.py",
               "--out_dir", DATA_DIR, "--hf_cache", HF_CACHE, "--smoke"]
        _run(cmd)
        _mark_done(PRETRAIN_TRAIN, DRIVE_DATA)
        _mark_done(PRETRAIN_DEV,   DRIVE_DATA)
        assert os.path.exists(PRETRAIN_TRAIN + ".done")
        assert os.path.exists(PRETRAIN_DEV   + ".done")
        _ok("pretrain-data", time.perf_counter() - t0, "ran + marked")
except Exception as e:
    _fail("pretrain-data", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4a: Second call — must skip (idempotency)
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 4a (second call — must skip)")
t0 = time.perf_counter()
try:
    assert _data_is_ready(PRETRAIN_TRAIN, DRIVE_DATA), "marker missing after first call"
    assert _data_is_ready(PRETRAIN_DEV,   DRIVE_DATA), "marker missing after first call"
    _ok("pretrain-data-idempotent", time.perf_counter() - t0, "skipped correctly")
except Exception as e:
    _fail("pretrain-data-idempotent", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Spider (smoke — skipped)
# ─────────────────────────────────────────────────────────────────────────────
_section("Spider (smoke mode — skipped)")
t0 = time.perf_counter()
try:
    assert SMOKE, "expected SMOKE=True"
    print("  Smoke mode — Spider not needed.")
    _ok("spider", time.perf_counter() - t0, "skipped (smoke)")
except Exception as e:
    _fail("spider", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6a: SFT data
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 6a: SFT data")
t0 = time.perf_counter()

SFT_TRAIN = os.path.join(DATA_DIR, "sft_train.h5")
SFT_DEV   = os.path.join(DATA_DIR, "sft_dev.h5")

try:
    if _data_is_ready(SFT_TRAIN, DRIVE_DATA) and _data_is_ready(SFT_DEV, DRIVE_DATA):
        _ok("sft-data", time.perf_counter() - t0, "skipped — .done present")
    else:
        cmd = [sys.executable, "scripts/prepare_sft_data.py",
               "--out_dir", DATA_DIR, "--hf_cache", HF_CACHE, "--smoke"]
        _run(cmd)
        _mark_done(SFT_TRAIN, DRIVE_DATA)
        _mark_done(SFT_DEV,   DRIVE_DATA)
        assert os.path.exists(SFT_TRAIN + ".done")
        _ok("sft-data", time.perf_counter() - t0, "ran + marked")
except Exception as e:
    _fail("sft-data", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7a: RL prompts
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 7a: RL prompts")
t0 = time.perf_counter()

RL_TRAIN = os.path.join(DATA_DIR, "sql_prompts_train.jsonl")
RL_DEV   = os.path.join(DATA_DIR, "sql_prompts_dev.jsonl")

try:
    if _data_is_ready(RL_TRAIN, DRIVE_DATA) and _data_is_ready(RL_DEV, DRIVE_DATA):
        _ok("rl-prompts", time.perf_counter() - t0, "skipped — .done present")
    else:
        cmd = [sys.executable, "scripts/prepare_rl_prompts.py",
               "--out_dir", DATA_DIR, "--smoke"]
        _run(cmd)
        _mark_done(RL_TRAIN, DRIVE_DATA)
        _mark_done(RL_DEV,   DRIVE_DATA)
        assert os.path.exists(RL_TRAIN + ".done")
        _ok("rl-prompts", time.perf_counter() - t0, "ran + marked")
except Exception as e:
    _fail("rl-prompts", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Corrupt checkpoint recovery test
# ─────────────────────────────────────────────────────────────────────────────
_section("Corrupt checkpoint recovery")
t0 = time.perf_counter()

FAKE_CKPT = os.path.join(CKPT_DIR, "_corrupt_test.pt")
try:
    with open(FAKE_CKPT, "wb") as f:
        f.write(b"garbage bytes that are not a valid pytorch checkpoint")
    assert os.path.exists(FAKE_CKPT), "file not created"
    assert not _ckpt_is_valid(FAKE_CKPT), "_ckpt_is_valid should return False for garbage"
    nr, res = _training_needed(FAKE_CKPT, "pretrain", 20)
    assert nr is True,  "needs_run should be True after corrupt"
    assert res is False, "should_resume should be False after corrupt"
    assert not os.path.exists(FAKE_CKPT), "corrupt file should have been deleted"
    _ok("corrupt-recovery", time.perf_counter() - t0, "detected + removed")
except Exception as e:
    _fail("corrupt-recovery", time.perf_counter() - t0, str(e))
    if os.path.exists(FAKE_CKPT):
        os.remove(FAKE_CKPT)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Pretrain (should skip — checkpoint complete)
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 5: Pretrain")
t0 = time.perf_counter()
try:
    assert os.path.exists(PRETRAIN_TRAIN), "pretrain data missing"
    nr, res = _training_needed(PRETRAIN_CKPT, "pretrain", PRETRAIN_TOTAL)
    if not nr:
        _ok("pretrain-train", time.perf_counter() - t0, "skipped — complete")
    else:
        cmd = [sys.executable, "scripts/pretrain_base.py", "--config", CONFIG_PRETRAIN]
        if res:
            cmd += ["--resume", "latest"]
        _run(cmd)
        assert _ckpt_is_valid(PRETRAIN_CKPT, "pretrain"), "checkpoint invalid after training"
        _ok("pretrain-train", time.perf_counter() - t0, "ran")
except Exception as e:
    _fail("pretrain-train", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6b: SFT (should skip — checkpoint complete)
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 6b: SFT")
t0 = time.perf_counter()
try:
    assert _ckpt_is_valid(PRETRAIN_CKPT, "pretrain"), "pretrain checkpoint missing"
    assert os.path.exists(SFT_TRAIN), "sft data missing"
    nr, res = _training_needed(SFT_CKPT, "sft", sft_mode=True)
    if not nr:
        _ok("sft-train", time.perf_counter() - t0, "skipped — complete")
    else:
        cmd = [sys.executable, "scripts/train_sft.py", "--config", CONFIG_SFT]
        if res:
            cmd += ["--resume", "latest"]
        _run(cmd)
        assert _ckpt_is_valid(SFT_CKPT, "sft"), "checkpoint invalid after training"
        _ok("sft-train", time.perf_counter() - t0, "ran")
except Exception as e:
    _fail("sft-train", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7b: GRPO (should skip — checkpoint complete)
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 7b: GRPO")
t0 = time.perf_counter()
try:
    assert _ckpt_is_valid(SFT_CKPT, "sft"), "sft checkpoint missing"
    assert os.path.exists(RL_TRAIN), "rl prompts missing"
    nr, res = _training_needed(GRPO_CKPT, "grpo", GRPO_TOTAL)
    if not nr:
        _ok("grpo-train", time.perf_counter() - t0, "skipped — complete")
    else:
        cmd = [sys.executable, "scripts/train_grpo.py", "--config", CONFIG_GRPO]
        if res:
            cmd += ["--resume", "latest"]
        _run(cmd)
        assert _ckpt_is_valid(GRPO_CKPT, "grpo"), "checkpoint invalid after training"
        _ok("grpo-train", time.perf_counter() - t0, "ran")
except Exception as e:
    _fail("grpo-train", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Eval
# ─────────────────────────────────────────────────────────────────────────────
_section("Stage 8: Eval")
t0 = time.perf_counter()

SMOKE_DB      = os.path.join(DATA_DIR, "smoke.sqlite")
SMOKE_PROMPTS = os.path.join(DATA_DIR, "sql_prompts_train.jsonl")

try:
    cmd = [
        sys.executable, "scripts/eval_sql.py",
        "--config", CONFIG_BASE,
        "--max_new_tokens", "32",
        "--temperature", "0.0",
    ]
    for ckpt_path, stage in [(PRETRAIN_CKPT, "pretrain"),
                              (SFT_CKPT,      "sft"),
                              (GRPO_CKPT,     "grpo")]:
        if _ckpt_is_valid(ckpt_path, stage):
            cmd += ["--ckpt", ckpt_path]
    cmd += ["--smoke", "--smoke_db", SMOKE_DB, "--smoke_prompts", SMOKE_PROMPTS]
    _run(cmd)
    _ok("eval", time.perf_counter() - t0)
except Exception as e:
    _fail("eval", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Drive restore simulation (no actual Drive, but test the logic path)
# ─────────────────────────────────────────────────────────────────────────────
_section("Drive restore simulation")
t0 = time.perf_counter()
try:
    FAKE_DRIVE = "/tmp/fake_drive_sql_lm"
    FAKE_DATA  = "/tmp/fake_local_data"
    os.makedirs(FAKE_DRIVE, exist_ok=True)
    os.makedirs(FAKE_DATA,  exist_ok=True)

    # Simulate: file + marker on Drive, local is absent
    fake_h5 = os.path.join(FAKE_DATA, "test.h5")
    fake_h5_drive  = os.path.join(FAKE_DRIVE, "test.h5")
    fake_h5_marker = os.path.join(FAKE_DRIVE, "test.h5.done")
    with open(fake_h5_drive, "w") as f: f.write("fake hdf5 content")
    open(fake_h5_marker, "w").close()

    assert not os.path.exists(fake_h5), "should not exist locally yet"
    result = _data_is_ready(fake_h5, FAKE_DRIVE)
    assert result, "_data_is_ready should restore from Drive"
    assert os.path.exists(fake_h5), "file should now exist locally"
    assert os.path.exists(fake_h5 + ".done"), "marker should now exist locally"

    # Second call should hit local fast path
    result2 = _data_is_ready(fake_h5, FAKE_DRIVE)
    assert result2, "second call should return True"

    shutil.rmtree(FAKE_DRIVE)
    shutil.rmtree(FAKE_DATA)
    _ok("drive-restore", time.perf_counter() - t0)
except Exception as e:
    _fail("drive-restore", time.perf_counter() - t0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  Stage 9 smoke test results")
print(f"{'═'*60}")
n_ok   = sum(1 for _, s, _ in results if s == "ok")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
for cell, status, elapsed in results:
    mark = "✓" if status == "ok" else "✗"
    print(f"  {mark} {cell:35s} {elapsed:5.1f}s")
print(f"{'─'*60}")
print(f"  {n_ok} passed  {n_fail} failed")
print(f"{'═'*60}")

if n_fail:
    sys.exit(1)
