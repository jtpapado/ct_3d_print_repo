#!/bin/bash
# Download open fracture datasets (all CC BY 4.0):
#   fracatlas  0.34 GB  X-rays with fracture masks           doi:10.6084/m9.figshare.22363012
#   platif     6.05 GB  tibial plateau fracture X-rays + CT   doi:10.5281/zenodo.18007397
#   pengwin    7.61 GB  fractured pelvis CT + fragment labels doi:10.5281/zenodo.10927452
# usage: download_fractures.sh [fracatlas|platif|pengwin|all] [root_dir]
set -u
WHAT="${1:-all}"
ROOT="${2:-$HOME/fracture_datasets}"
Z=https://zenodo.org/api/records
get() { wget -c -q --show-progress --progress=dot:giga -O "$1" "$2" || echo "FAILED: $2"; }

if [[ $WHAT == fracatlas || $WHAT == all ]]; then
  D="$ROOT/fracatlas"; mkdir -p "$D/extracted"
  get "$D/FracAtlas.zip" "https://ndownloader.figshare.com/files/65518038"
  unzip -q -n "$D/FracAtlas.zip" -d "$D/extracted"
fi

if [[ $WHAT == platif || $WHAT == all ]]; then
  D="$ROOT/platif"; mkdir -p "$D/extracted"
  get "$D/extracted/Tibial_Plateau_Fracture_Metadata.xlsx" "$Z/18007397/files/Tibial%20Plateau%20Fracture%20Metadata.xlsx/content"
  for i in 1 2 3 4 5; do
    get "$D/Patient_Data_Part_$i.zip" "$Z/18007397/files/Patient%20Data_Part%20$i.zip/content"
    unzip -q -n "$D/Patient_Data_Part_$i.zip" -d "$D/extracted"
  done
fi

if [[ $WHAT == pengwin || $WHAT == all ]]; then
  # images and labels use the same file names (001.mha ...), so keep them in separate folders
  D="$ROOT/pengwin_ct"; mkdir -p "$D/images" "$D/labels"
  get "$D/PENGWIN_CT_train_labels.zip" "$Z/10927452/files/PENGWIN_CT_train_labels.zip/content"
  unzip -q -n "$D/PENGWIN_CT_train_labels.zip" -d "$D/labels"
  for p in 1 2; do
    get "$D/PENGWIN_CT_train_images_part$p.zip" "$Z/10927452/files/PENGWIN_CT_train_images_part$p.zip/content"
    unzip -q -n "$D/PENGWIN_CT_train_images_part$p.zip" -d "$D/images"
  done
fi
du -sh "$ROOT"/* 2>/dev/null
