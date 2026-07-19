#!/usr/bin/env bash
# Fix the HuggingFace cache on the DGX Spark being owned by root.
#
# Symptom:
#   OSError: PermissionError at ~/.cache/huggingface/hub/models--<...>
#   when downloading <model>. Check cache directory permissions.
#
# Cause (observed 2026-07-18):
#   drwxr-xr-x 4 root root  /home/mhough/.cache/huggingface/hub   [created Jul  1]
#   Something ran as root -- most likely a container mounting $HOME -- and created
#   the cache tree with root ownership. Every subsequent user-level model load
#   fails, including the Flax-vs-PyTorch RoPE comparison in docs/flax_rope_bug.md.
#
# This only chowns files ALREADY owned by root under that path. It does not
# delete anything and does not touch the downloaded blobs' contents, so any
# models already cached stay cached.
#
# Run ON THE SPARK:  bash scripts/fix_hf_cache_perms.sh

set -euo pipefail

CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
ME="$(id -un)"
GRP="$(id -gn)"

if [[ ! -d "$CACHE" ]]; then
    echo "no cache at $CACHE -- nothing to do"
    exit 0
fi

echo "cache:  $CACHE"
echo "user:   $ME:$GRP"
echo
echo "currently root-owned entries (top 10):"
find "$CACHE" -maxdepth 2 ! -user "$ME" -printf '  %u:%g  %p\n' 2>/dev/null | head -10 || true

N=$(find "$CACHE" ! -user "$ME" 2>/dev/null | wc -l)
echo
echo "$N entries not owned by $ME."
if [[ "$N" -eq 0 ]]; then
    echo "nothing to fix."
    exit 0
fi

read -r -p "chown -R $ME:$GRP $CACHE ? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 1; }

sudo chown -R "$ME:$GRP" "$CACHE"
# Group-writable so a future container run as a different member does not
# recreate the problem quite so easily.
sudo chmod -R u+rwX,g+rwX "$CACHE"

echo
echo "done. verifying:"
ls -ld "$CACHE/hub"
REMAIN=$(find "$CACHE" ! -user "$ME" 2>/dev/null | wc -l)
echo "entries still not owned by $ME: $REMAIN"

echo
echo "NOTE: /  is at 86% (126G free). The DeepSeek-R1-Distill-Llama-8B checkpoint"
echo "is ~16GB in bf16; Gemma 3 27B is ~54GB. Check headroom before large pulls:"
echo "    df -h ~/.cache"
