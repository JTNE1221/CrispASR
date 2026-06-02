"""
CrispASR — mimo-asr GPU repro + diagnostic (PLAN #115 option C)

The CPU path works (sibling kernel crispasr-mimo-asr-cpu-validate, JFK PASS).
The GPU path is broken two ways (PLAN #115):
  - weights on GPU + GPU compute (commit 89111260): silent-empty, exit 0.
  - weights on CPU + GPU compute (the §56 config): `buffer is nil`.
Option A force-pins everything to CPU (correct, slow). Option C is the
proper GPU graph fix — but it can't be debugged on the local M1 (4.2 GB
model, box memory-saturated). This kernel runs it on a Kaggle GPU box.

What it does on a GPU notebook:
  1. CUDA build of crispasr-cli (+ crispasr-diff for the stage self-diff).
  2. download mimo-asr-q4_k.gguf + tokenizer from cstr/.
  3. run mimo-asr on JFK two ways:
       cpu  — default (force-CPU, option A). Expect PASS (reference text).
       gpu  — CRISPASR_MIMO_FORCE_GPU=1 (weights+compute on GPU, option C
              config). Expect EMPTY/WRONG — this is the bug, on CUDA this
              time (the M1 Metal and Blackwell symptoms differed, so
              confirming CUDA repro matters).
  4. dump the full stderr of the GPU run (verbosity 2) so the failure
     point — `buffer is nil`, NaN logits, or immediate EOS — is visible
     in the logs and localises the next step.

Reference (HISTORY §56) JFK transcript: "...ask not what your country can
do for you. Ask what you can do for your country."

All build/report plumbing comes from the shared harness
tools/kaggle/kaggle_harness.py (kh.*). Set enable_gpu=true in the kernel
metadata (see kernel-metadata-gpu.json).
"""

import os
import re
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
SAMPLE = WORK / "jfk.wav"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")


def _sh_preclone(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print("[pre-clone] cloning CrispASR for shared harness", flush=True)
if not REPO.exists():
    _sh_preclone(
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}"
    )
else:
    _sh_preclone(f"git -C {REPO} fetch --depth 1 origin {CRISPASR_REF} && "
                 f"git -C {REPO} reset --hard FETCH_HEAD")

import sys
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

if kh.resolve_hf_token():
    print("HF auth: token loaded (progress mirror enabled)", flush=True)
else:
    print("HF auth: anonymous (progress mirror disabled; local JSONL only)", flush=True)

kh.step("script.start")
EXPECTED_JFK = "ask not what your country can do for you"

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("clone.done", sha=sha, ref=CRISPASR_REF)

# ── CUDA build ───────────────────────────────────────────────────────────
kh.step("build.begin")
BUILD.mkdir(exist_ok=True)
kh.install_build_toolchain()
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja "
    "-DCMAKE_BUILD_TYPE=Release "
    "-DBUILD_SHARED_LIBS=ON "
    "-DCRISPASR_BUILD_TESTS=OFF "
    "-DGGML_CUDA=ON "
    "-DCMAKE_CUDA_ARCHITECTURES=native "
    + " ".join(kh.cache_and_link_flags())
)
njobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
kh.step("build.configured")
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}")
CRISPASR = BUILD / "bin" / "crispasr"
assert CRISPASR.is_file(), f"crispasr binary missing at {CRISPASR}"
kh.step("build.done", binary=str(CRISPASR))

# ── download model + tokenizer ───────────────────────────────────────────
kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download

for repo_id, fname in [
    ("cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf"),
    ("cstr/mimo-tokenizer-GGUF", "mimo-tokenizer-q4_k.gguf"),
]:
    kh.step(f"download.{fname}.begin", repo=repo_id)
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    kh.step(f"download.{fname}.done", path=p, size_mib=Path(p).stat().st_size // (1 << 20))

mimo_path = MODELS / "mimo-asr-q4_k.gguf"
subprocess.run(["cp", f"{REPO}/samples/jfk.wav", str(SAMPLE)], check=False)
assert SAMPLE.is_file(), "jfk.wav missing"
kh.step("download.done")

# ── run mimo-asr on JFK: CPU baseline + GPU repro ────────────────────────
def run_mimo(label: str, env_extra: dict, verbose: bool) -> tuple[str, str, bool]:
    out_stem = WORK / f"mimo-jfk-{label}"
    for ext in [".txt", ".srt"]:
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [str(CRISPASR), "-m", str(mimo_path), "--backend", "mimo-asr",
           "-f", str(SAMPLE), "-of", str(out_stem), "-otxt"]
    if not verbose:
        cmd.append("-np")
    kh.step(f"run.{label}.begin", cmd=" ".join(cmd), env=list(env_extra.keys()))
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, "MIMO_ASR_BENCH": "1", **env_extra},
                       capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    log = (r.stdout or "") + (r.stderr or "")
    txt_path = out_stem.with_suffix(".txt")
    has_file = txt_path.exists() and txt_path.stat().st_size > 0
    text = txt_path.read_text().strip() if has_file else ""
    ok = EXPECTED_JFK in text.lower()
    kh.step(f"run.{label}.done", exit=r.returncode, wall_s=round(dt, 1),
            has_output=has_file, chars=len(text), ok=ok)
    return text, log, ok


kh.step("run.section.begin")
cpu_text, cpu_log, cpu_ok = run_mimo("cpu", {}, verbose=False)
# GPU: weights + compute on GPU (option C config), verbose so the failure
# point shows in the log.
gpu_text, gpu_log, gpu_ok = run_mimo("gpu", {"CRISPASR_MIMO_FORCE_GPU": "1"}, verbose=True)

# ── Summary ──────────────────────────────────────────────────────────────
def status(text, ok):
    return "PASS (matches ref)" if ok else ("EMPTY (no output)" if not text else "WRONG (differs from ref)")

print("\n" + "=" * 72, flush=True)
print(f"SUMMARY — mimo-asr JFK on {sha[:8]} — CUDA kernel (PLAN #115 option C)", flush=True)
print("=" * 72, flush=True)
print(f"  CPU (force-CPU, option A):        {status(cpu_text, cpu_ok)}", flush=True)
print(f"  GPU (CRISPASR_MIMO_FORCE_GPU=1):  {status(gpu_text, gpu_ok)}", flush=True)
print(f"  cpu text: {cpu_text[:160]}", flush=True)
print(f"  gpu text: {gpu_text[:160]}", flush=True)
print("\n--- GPU run stderr (last 60 lines — failure point) ---", flush=True)
for line in gpu_log.splitlines()[-60:]:
    print(line, flush=True)

kh.step("summary", cpu=status(cpu_text, cpu_ok), gpu=status(gpu_text, gpu_ok),
        gpu_tail="\n".join(gpu_log.splitlines()[-12:]), sha=sha)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
