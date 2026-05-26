"""
CrispASR — mimo-asr CPU-path validation (PLAN #115)

Question: on current main, mimo-asr produces zero output on JFK (11 s) on
M1 Metal and segfaults on a 5 min clip. The smoking-gun commit by
inspection is `89111260` ("perf #72: load weights to GPU when use_gpu=true")
which flipped `core_gguf::load_weights(..., ctx->backend_cpu, ...)` to
`..., ctx->backend, ...`. That commit's own message foresees the regression:
"If a platform regresses, add a CRISPASR_FORCE_CPU_WEIGHTS=1 escape hatch
— none seen yet." Now we have one.

We can't safely repro on the local M1 box (4.5 GB CPU mimo Q4_K + already
loaded benchmark sweep risks OOM). This kernel runs on a Kaggle CPU
notebook (separate quota from GPU, no queue starvation) and answers:

    Does HEAD mimo-asr produce a JFK transcript on a pure-CPU build?

If yes → bug is GPU-specific (PLAN #72 broke the GPU residency path only).
If no  → CPU path is also broken; more regressions stacked since HISTORY §56.

Reference (HISTORY §56): the JFK transcript should be verbatim
"And so, my fellow Americans, ask not what your country can do for you.
Ask what you can do for your country."

Patterns lifted from tools/kaggle/crispasr-regression.py (the rebake script
hardened across the 2026-05-25 v7→v13 fix train per
[[project_kaggle_rebake_fragilities]]):
  - step() / progress.jsonl + HF mirror at cstr/crispasr-kaggle-progress
  - sh_with_progress() Popen-based build streamer + ninja [X/N] parsing
  - build_heartbeat() ticker so cmake/ninja hangs show in progress.jsonl
  - HF_TOKEN read from the chr1str/crispasr-hf-token dataset (mounted via
    kernel-metadata.json:dataset_sources), with Kaggle Secrets fallback
"""

import contextlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Unbuffered I/O + progress.jsonl checkpointer ─────────────────────────
# Kaggle persists logs only at process end; a hang past stdout buffer fill
# is invisible until the kernel terminates. Force line-buffered stdio and
# JSONL-checkpoint every step so a hang shows the last completed step via
# `kaggle kernels output --file-pattern 'progress.jsonl'` (post-mortem)
# OR via the HF mirror dataset (mid-run, see below).
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
SAMPLE = WORK / "jfk.wav"

_PROGRESS = WORK / "progress.jsonl"
_T0 = time.time()
_HF_REPO = "cstr/crispasr-kaggle-progress"
_HF_PATH = (
    f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    f"-mimo-asr-cpu-vs-gpu.jsonl"
)
_HF_INTERVAL_S = 30.0
_HF_LAST = 0.0


def _push_progress_to_hf(force: bool = False) -> None:
    global _HF_LAST
    now = time.time()
    if not force and (now - _HF_LAST) < _HF_INTERVAL_S:
        return
    if not os.environ.get("HF_TOKEN"):
        return
    if not _PROGRESS.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(_PROGRESS),
            path_in_repo=_HF_PATH,
            repo_id=_HF_REPO,
            repo_type="dataset",
            commit_message=f"progress @ {now - _T0:.0f}s",
        )
        _HF_LAST = now
    except Exception:
        pass


def step(name: str, **extra) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name,
        **extra,
    }
    try:
        _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        with _PROGRESS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    print(f"[step {rec['elapsed_s']:>7.1f}s] {name}" +
          (f"  {extra}" if extra else ""), flush=True)
    _push_progress_to_hf()


# ── Build-step heartbeat + Popen line streamer ───────────────────────────
# Kaggle buffers parent stdout heavily; subprocess.check_call for cmake
# means ninja's output flows through that buffered pipe and a hang past
# the buffer fill is invisible. Same pattern as the rebake script: Popen
# with stdout=PIPE, iterate lines, print each with explicit flush. The
# heartbeat thread parses ninja's [X/N] + last TU so progress.jsonl ticks
# include "compile 208/360 t5_translate.cpp" instead of just elapsed.
_BUILD_PROGRESS: dict = {"last_ninja": None, "last_tu": None, "lines": 0}
_NINJA_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_TU_RE = re.compile(r"(\S+\.(?:cpp|cc|cxx|c|cu))(?::|\s|$)")


def sh_with_progress(cmd: str, cwd: Path | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            _BUILD_PROGRESS["lines"] += 1
            m = _NINJA_RE.match(line)
            if m:
                _BUILD_PROGRESS["last_ninja"] = f"{m.group(1)}/{m.group(2)}"
            m = _TU_RE.search(line)
            if m:
                _BUILD_PROGRESS["last_tu"] = m.group(1).rsplit("/", 1)[-1]
    finally:
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


@contextlib.contextmanager
def build_heartbeat(label: str, interval_s: float = 30.0):
    t_start = time.time()
    stop_event = threading.Event()

    def _ticker():
        while not stop_event.wait(interval_s):
            extra: dict = {"elapsed_in_block_s": round(time.time() - t_start, 1)}
            if _BUILD_PROGRESS["last_ninja"]:
                extra["ninja"] = _BUILD_PROGRESS["last_ninja"]
                extra["tu"] = _BUILD_PROGRESS["last_tu"]
                extra["lines"] = _BUILD_PROGRESS["lines"]
            step(f"{label}.heartbeat", **extra)

    thread = threading.Thread(target=_ticker, daemon=True, name=f"hb-{label}")
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


# ── HF auth: Kaggle Dataset first, Secrets API fallback ──────────────────
# [[project_kaggle_rebake_fragilities]] #5: UserSecretsClient.get_secret
# flakes with ConnectionError on batch kernels even when JWT/Attach are
# correct. Durable fix is the chr1str/crispasr-hf-token dataset mounted
# via kernel-metadata.json:dataset_sources.
def _read_hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (
        Path("/kaggle/input/crispasr-hf-token/hf_token.txt"),
        Path("/kaggle/input/crispasr-hf-token/HF_TOKEN"),
    ):
        if p.is_file():
            tok = p.read_text().strip()
            if tok:
                return tok
    try:
        from kaggle_secrets import UserSecretsClient
        for attempt in range(3):
            try:
                return UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                time.sleep(5 * (attempt + 1))
    except Exception:
        pass
    return None


_tok = _read_hf_token()
if _tok:
    os.environ["HF_TOKEN"] = _tok
    print("HF auth: token loaded (progress mirror enabled)", flush=True)
else:
    print("HF auth: anonymous (progress mirror disabled; local JSONL only)", flush=True)

step("script.start")

# Branch-parametrized so re-runs against fixes are one env var away.
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
EXPECTED_JFK = "ask not what your country can do for you"

# ── Step 1: clone ────────────────────────────────────────────────────────

step("clone.begin", ref=CRISPASR_REF)
if not REPO.exists():
    sh_with_progress(
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}"
    )
else:
    sh_with_progress(f"git -C {REPO} pull --ff-only")
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
step("clone.done", sha=sha)

# ── Step 2: CPU-only build ───────────────────────────────────────────────
# We are explicitly on a CPU kernel (enable_gpu=false in kernel-metadata)
# because: (a) GPU quota was queue-blocking us for 14+ h on the prior
# attempt; (b) the question we're answering — "does the CPU path produce
# any text at all on HEAD?" — doesn't need GPU. The full GPU vs CPU
# regression bisect can run when GPU quota frees, with this same script
# pointed at enable_gpu=true.

step("build.begin")
BUILD.mkdir(exist_ok=True)
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja "
    "-DCMAKE_BUILD_TYPE=Release "
    "-DBUILD_SHARED_LIBS=ON "
    "-DCRISPASR_BUILD_TESTS=OFF"
)
njobs = max(4, multiprocessing.cpu_count())
with build_heartbeat("cmake-configure"):
    sh_with_progress(cmake_cmd)
step("build.configured")
with build_heartbeat("cmake-build"):
    sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}"
    )
CRISPASR = BUILD / "bin" / "crispasr"
assert CRISPASR.is_file(), f"crispasr binary missing at {CRISPASR}"
step("build.done", binary=str(CRISPASR))

# ── Step 3: download mimo-asr + tokenizer ─────────────────────────────────

step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download

for repo_id, fname in [
    ("cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf"),
    ("cstr/mimo-tokenizer-GGUF", "mimo-tokenizer-q4_k.gguf"),
]:
    step(f"download.{fname}.begin", repo=repo_id)
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    step(f"download.{fname}.done", path=p, size_mib=Path(p).stat().st_size // (1 << 20))

mimo_path = MODELS / "mimo-asr-q4_k.gguf"
tok_path = MODELS / "mimo-tokenizer-q4_k.gguf"

# JFK sample from repo
subprocess.run(["cp", f"{REPO}/samples/jfk.wav", str(SAMPLE)], check=False)
assert SAMPLE.is_file(), "jfk.wav missing"
step("download.done")

# ── Step 4: run mimo-asr on JFK ──────────────────────────────────────────

def run_mimo(label: str, extra_args: list[str]) -> tuple[str, bool]:
    out_stem = WORK / f"mimo-jfk-{label}"
    for ext in [".txt", ".srt"]:
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [
        str(CRISPASR),
        "-m", str(mimo_path),
        "--backend", "mimo-asr",
        "-f", str(SAMPLE),
        "-of", str(out_stem),
        "-otxt",
        "-np",
    ] + extra_args
    step(f"run.{label}.begin", cmd=" ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, "MIMO_ASR_BENCH": "1"},
                       capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    log = (r.stdout or "") + (r.stderr or "")
    bench = re.search(r"mimo_asr_bench:.*", log)
    txt_path = out_stem.with_suffix(".txt")
    has_file = txt_path.exists() and txt_path.stat().st_size > 0
    text = txt_path.read_text().strip() if has_file else ""
    ok = EXPECTED_JFK in text.lower()
    step(
        f"run.{label}.done",
        exit=r.returncode,
        wall_s=round(dt, 1),
        bench=bench.group(0) if bench else None,
        has_output=has_file,
        chars=len(text),
        ok=ok,
    )
    if not has_file or not text:
        # Surface tail of stderr to logs for diagnosis
        print("--- last 30 lines of stderr ---", flush=True)
        for line in log.splitlines()[-30:]:
            print(line, flush=True)
    return text, ok


step("run.section.begin")
text, ok = run_mimo("cpu", [])

# ── Summary ──────────────────────────────────────────────────────────────

step("summary", text=text[:300], ok=ok, sha=sha)
print("\n" + "=" * 70)
print(f"SUMMARY — mimo-asr JFK on HEAD ({sha[:8]}) — CPU-only kernel")
print("=" * 70)
status = "PASS (matches reference)" if ok else ("EMPTY (no output)" if not text else "WRONG (text differs from reference)")
print(f"  result: {status}")
print(f"  text:   {text[:200]}")
print()
print("Interpretation:")
if ok:
    print("  CPU path works on HEAD. The mimo-asr regression observed on M1 Metal")
    print("  is GPU-specific (PLAN #72 commit 89111260 'load weights to GPU').")
    print("  Fix: either revert the one-line backend_cpu→backend swap, or fix the")
    print("  GPU tensor routing in mimo_asr_build_prefill_graph.")
elif text:
    print("  Output produced but doesn't match the reference. Could be a separate")
    print("  text bug (detokenizer / stop-token handling) or different model.")
else:
    print("  No output even on CPU. There are more regressions stacked on top of")
    print("  #72. Bisect further: HISTORY §56 (dae361f2) was the last known good.")

_push_progress_to_hf(force=True)
step("script.end")
