#!/bin/bash
# Download two high-resolution cadaveric knee CTs (~8.2 GB) from the
# University of Leeds IMBE Knee Dataset (CC BY 4.0, doi:10.5518/826).
set -u
ROOT="${1:-$HOME/knee_ct_datasets}"
L="https://archive.researchdata.leeds.ac.uk"
mkdir -p "$ROOT/leeds_LTKN7607" "$ROOT/leeds_LTKN8941"

get() { wget -c -q --show-progress --progress=dot:giga -O "$1" "$2" || echo "FAILED: $2"; }

get "$ROOT/leeds_LTKN7607/LTKN7607_readme.txt"      "$L/1598/9/LTKN7607_readme.txt"
get "$ROOT/leeds_LTKN7607/LTKN7607_geom.stl"        "$L/1598/8/LTKN7607_geom.stl"
get "$ROOT/leeds_LTKN7607/LTKN7607_CT_intact.zip"   "$L/1598/2/LTKN7607_CT_intact.zip"
get "$ROOT/leeds_LTKN8941/LTKN8941_readme.txt"      "$L/1593/8/LTKN8941_readme.txt"
get "$ROOT/leeds_LTKN8941/LTKN8941_geom.stl"        "$L/1593/7/LTKN8941_geom.stl"
get "$ROOT/leeds_LTKN8941/LTKN8941_CT_dissected.zip" "$L/1593/1/LTKN8941_CT_dissected.zip"

for k in LTKN7607 LTKN8941; do
  mkdir -p "$ROOT/leeds_$k/ct"
  unzip -q -n "$ROOT"/leeds_$k/${k}_CT_*.zip -d "$ROOT/leeds_$k/ct" && echo "unzipped $k"
done
