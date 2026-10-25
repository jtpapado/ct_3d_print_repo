#!/bin/bash
# Download the VSDFullBodyBoneModels database (0.18 GB): closed surface meshes of the
# lower-body bones of 30 subjects, from CT. CC BY-NC-SA 4.0, doi:10.5281/zenodo.8316967
set -u
D="${1:-$HOME/knee_ct_datasets/vsd_bone_models}"
mkdir -p "$D"
wget -c -q --show-progress -O "$D/VSDFullBodyBoneModels-v3.0.zip" \
  "https://zenodo.org/api/records/8316967/files/MCM-Fischer/VSDFullBodyBoneModels-v3.0.zip/content"
unzip -q -n "$D/VSDFullBodyBoneModels-v3.0.zip" -d "$D/extracted"
ls -d "$D"/extracted/*/Bones
