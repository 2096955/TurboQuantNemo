#!/usr/bin/env bash
# Two-stage quantize: dequantize Kimi K2.6 source to bf16, then re-quantize
# with 2-bit routed experts (4-bit shared/dense/attn).
#
# RECOMMENDED ALTERNATIVE: the in-memory one-stage path skips the bf16
# intermediate entirely and saves ~1.9 TB of peak disk. Run instead:
#
#     PYTHONPATH=mlx-lm python3 scripts/convert_kimi_2bit_chunked.py \
#         --src /Volumes/Samsung9904tb/Kimi-K2.6 \
#         --dst /Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts \
#         --source-quantized
#
# The --source-quantized flag dequantizes the source in-memory (lazy
# mx.dequantize, never materialised to disk), then runs the same chunked
# requantize + per-tensor save loop as the two-stage path.
#
# This wrapper (two-stage) is preserved for cases where you want a
# reusable bf16 intermediate or are debugging the dequantize step
# independently. Stage 2 calls convert_kimi_2bit_chunked.py to avoid the
# kIOGPUCommandBufferCallbackErrorTimeout that the standard
# mlx_lm.convert hits on Kimi K2.6 (per the 2026-05-01 phase-2-quantize
# format-gap handover).
#
# Why two stages exist at all: the source checkpoint already has
# compressed-tensors 4-bit routed experts. mlx_lm.convert's
# --mixed-expert-bits flag is silently no-op'd on already-quantized
# layers (utils.py quantize_model only requantizes via nn.quantize, which
# skips already-QuantizedLinear modules). Dequantizing first makes the
# predicate fire.
#
# Stage 1 (dequantize): ~1.9 TB intermediate, 30-90 min wall.
# Stage 2 (requantize):  ~302 GB final, 30-90 min wall.
# Intermediate is deleted after Stage 2 completes successfully.
#
# Total disk peak during Stage 2: source 554 GB + intermediate 1.9 TB +
# final ~302 GB in flight = ~2.76 TB. SSD must have that much free at
# start, or use the one-stage --source-quantized path.

set -euo pipefail

SRC="${KIMI_SRC:-/Volumes/Samsung9904tb/Kimi-K2.6}"
DST="${KIMI_DST:-/Volumes/Samsung9904tb/Kimi-K2.6-2bit-experts}"
INTERMEDIATE="${KIMI_INTERMEDIATE:-/Volumes/Samsung9904tb/Kimi-K2.6-bf16}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"  # set to 1 to skip intermediate cleanup

# Volume guard — Kimi conversions must target /Volumes/Samsung9904tb only.
# Internal disk cannot fit the 1.1 TB bf16 intermediate, and a phantom mount
# would silently misroute multi-TB writes. Verify the volume identity before
# any path validation. Set KIMI_VOLUME_GUARD=0 to bypass (not recommended).
KIMI_VOLUME_GUARD="${KIMI_VOLUME_GUARD:-1}"
KIMI_REQUIRED_VOLUME="/Volumes/Samsung9904tb"

if [ "$KIMI_VOLUME_GUARD" = "1" ]; then
    for path in "$SRC" "$DST" "$INTERMEDIATE"; do
        case "$path" in
            "$KIMI_REQUIRED_VOLUME"|"$KIMI_REQUIRED_VOLUME"/*) ;;
            *)
                echo "ERROR: $path is not under $KIMI_REQUIRED_VOLUME." >&2
                echo "  Kimi conversions must target the Samsung 4 TB SSD only." >&2
                echo "  Set KIMI_VOLUME_GUARD=0 to bypass (you almost certainly should not)." >&2
                exit 1
                ;;
        esac
    done
    if [ ! -d "$KIMI_REQUIRED_VOLUME" ]; then
        echo "ERROR: $KIMI_REQUIRED_VOLUME is not mounted." >&2
        exit 1
    fi
    DISKUTIL_INFO=$(diskutil info "$KIMI_REQUIRED_VOLUME" 2>/dev/null || true)
    if ! echo "$DISKUTIL_INFO" | grep -q "Volume Name:.*Samsung9904tb"; then
        echo "ERROR: $KIMI_REQUIRED_VOLUME does not report Volume Name 'Samsung9904tb'." >&2
        echo "  Possible phantom mount. Run 'diskutil info $KIMI_REQUIRED_VOLUME' and resolve before retrying." >&2
        exit 1
    fi
    if ! echo "$DISKUTIL_INFO" | grep -q "File System Personality:.*APFS"; then
        echo "ERROR: $KIMI_REQUIRED_VOLUME is not APFS as expected." >&2
        exit 1
    fi
fi

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

# Disk space check — need ~2.8 TB peak (source 554 GB + bf16 ~1.9 TB +
# 2bit ~302 GB during stage 2). Use the one-stage --source-quantized path
# in convert_kimi_2bit_chunked.py if you don't have that much headroom.
AVAIL_KB=$(df -k "$(dirname "$DST")" | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 2800 ]; then
    echo "ERROR: only ${AVAIL_GB} GB free at $(dirname "$DST"); need >= 2800 GB for the two-stage convert (peak: source + bf16 intermediate + final)." >&2
    echo "  Consider the one-stage path:" >&2
    echo "    PYTHONPATH=mlx-lm python3 scripts/convert_kimi_2bit_chunked.py \\" >&2
    echo "        --src $SRC --dst $DST --source-quantized" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

echo
echo "================================================================"
echo "STAGE 1: dequantize source -> bf16 intermediate"
echo "  source:       $SRC (~554 GB)"
echo "  intermediate: $INTERMEDIATE (~1.9 TB target)"
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
echo "STAGE 2: re-quantize bf16 -> 2-bit routed experts (chunked)"
echo "  intermediate: $INTERMEDIATE"
echo "  final:        $DST (~302 GB target)"
echo "  Note: uses convert_kimi_2bit_chunked.py to avoid the GPU"
echo "        watchdog timeout that bites the standard mlx_lm.convert"
echo "        on Kimi K2.6 (per 2026-05-01 handover)."
echo "================================================================"
echo

STAGE2_START=$(date +%s)
PYTHONPATH=mlx-lm python3 scripts/convert_kimi_2bit_chunked.py \
    --src "$INTERMEDIATE" \
    --dst "$DST" \
    --mixed-expert-bits 2 \
    --shared-expert-bits 4 \
    --q-bits 4 \
    --q-group-size 64 \
    --q-mode affine

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
