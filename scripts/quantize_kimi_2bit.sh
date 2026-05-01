#!/usr/bin/env bash
# Two-stage quantize: dequantize Kimi K2.6 source to bf16, then re-quantize
# with 2-bit routed experts (4-bit shared/dense/attn).
#
# Why two stages: the source checkpoint already has compressed-tensors 4-bit
# routed experts. mlx_lm.convert's --mixed-expert-bits flag is silently
# no-op'd on already-quantized layers (utils.py quantize_model only
# requantizes via nn.quantize, which skips already-QuantizedLinear modules).
# The fix is to dequantize first, then re-apply the 2-bit predicate on the
# resulting bf16 weights.
#
# Stage 1 (dequantize): ~1.1 TB intermediate, 30-90 min wall.
# Stage 2 (requantize):  ~285 GB final, 30-90 min wall.
# Intermediate is deleted after Stage 2 completes successfully.
#
# Total disk peak during Stage 2: source 554 GB + intermediate 1.1 TB +
# final ~285 GB in flight = ~2.0 TB. SSD must have >= 2.0 TB free at start.

set -euo pipefail

SRC="${KIMI_SRC:-/Volumes/Samsung9904tb/Kimi-K2.6}"
DST="${KIMI_DST:-/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts}"
INTERMEDIATE="${KIMI_INTERMEDIATE:-/Volumes/Samsung9904tb/Kimi-K2.6-bf16}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"  # set to 1 to skip intermediate cleanup

if [ ! -d "$SRC" ]; then
    echo "ERROR: source not found: $SRC" >&2
    exit 1
fi

if [ -d "$DST" ]; then
    echo "ERROR: final destination already exists: $DST"
    echo "  Remove it first if you want to re-run:"
    echo "    rm -rf '$DST'"
    exit 1
fi

if [ -d "$INTERMEDIATE" ]; then
    echo "ERROR: intermediate dir already exists: $INTERMEDIATE"
    echo "  Either remove it (rm -rf '$INTERMEDIATE') or set KIMI_INTERMEDIATE to a different path."
    exit 1
fi

# Disk space check — need ~2.0 TB peak (source + bf16 + 2bit during stage 2)
AVAIL_KB=$(df -k "$(dirname "$DST")" | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 2000 ]; then
    echo "ERROR: only ${AVAIL_GB} GB free at $(dirname "$DST"); need >= 2000 GB for the two-stage convert (peak: source + bf16 intermediate + final)" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

echo
echo "================================================================"
echo "STAGE 1: dequantize source -> bf16 intermediate"
echo "  source:       $SRC (~554 GB)"
echo "  intermediate: $INTERMEDIATE (~1.1 TB target)"
echo "================================================================"
echo

STAGE1_START=$(date +%s)
PYTHONPATH=mlx-lm python3 -m mlx_lm convert \
    --hf-path "$SRC" \
    --mlx-path "$INTERMEDIATE" \
    --dequantize \
    --trust-remote-code

STAGE1_END=$(date +%s)
STAGE1_MIN=$(( (STAGE1_END - STAGE1_START) / 60 ))
echo
echo "Stage 1 complete in ${STAGE1_MIN} min. Intermediate size:"
du -sh "$INTERMEDIATE"

echo
echo "================================================================"
echo "STAGE 2: re-quantize bf16 -> 2-bit routed experts"
echo "  intermediate: $INTERMEDIATE"
echo "  final:        $DST (~285 GB target)"
echo "================================================================"
echo

STAGE2_START=$(date +%s)
PYTHONPATH=mlx-lm python3 -m mlx_lm convert \
    --hf-path "$INTERMEDIATE" \
    --mlx-path "$DST" \
    --quantize \
    --q-bits 4 \
    --q-group-size 64 \
    --q-mode affine \
    --mixed-expert-bits 2 \
    --shared-expert-bits 4 \
    --trust-remote-code

STAGE2_END=$(date +%s)
STAGE2_MIN=$(( (STAGE2_END - STAGE2_START) / 60 ))
echo
echo "Stage 2 complete in ${STAGE2_MIN} min. Final size:"
du -sh "$DST"

if [ "$KEEP_INTERMEDIATE" != "1" ]; then
    echo
    echo "Removing intermediate ${INTERMEDIATE} (set KEEP_INTERMEDIATE=1 to skip)..."
    rm -rf "$INTERMEDIATE"
    echo "Intermediate removed. Final disk free:"
    df -h "$(dirname "$DST")" | head -2
fi

echo
echo "================================================================"
echo "DONE. Final 2-bit checkpoint: $DST"
echo "  Stage 1 (dequantize):  ${STAGE1_MIN} min"
echo "  Stage 2 (requantize):  ${STAGE2_MIN} min"
echo "  Total:                 $(( STAGE1_MIN + STAGE2_MIN )) min"
echo "================================================================"
