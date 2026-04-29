#!/usr/bin/env bash
# Soft check for ~2.1 TB free on a chosen volume (Kimi staging). Human still must confirm.
# Usage: KIMI_STAGING_VOLUME=/Volumes/MySSD bash scripts/check_kimi_disk_prereq.sh
set -euo pipefail
MIN_TB="${KIMI_MIN_FREE_TB:-2.1}"
VOL="${KIMI_STAGING_VOLUME:-.}"

echo "Checking free space on: $VOL (minimum ${MIN_TB} TB recommended for Kimi K2.5 staging)"
df -h "$VOL"
avail_kb="$(df -k "$VOL" | awk 'NR==2 {print $4}')"
avail_tb="$(awk -v k="$avail_kb" 'BEGIN { printf "%.2f", k/1024/1024/1024 }')"
echo "Available (approx): ${avail_tb} TB"

awk -v avail="$avail_tb" -v min="$MIN_TB" 'BEGIN {
  if (avail+0 < min+0) { print "WARNING: below " min " TB — expand disk or choose another volume."; exit 1 }
  print "OK: at or above " min " TB (soft gate only — operator must still confirm staging policy)."; exit 0
}'
