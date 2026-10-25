"""Open a DICOM CT series with pydicom and render orthogonal views.

Loads every `step`-th slice and downsamples in-plane by the same factor so the
volume fits in memory, then saves preview images and a NIfTI copy.

usage: python open_knee_ct.py <dicom_dir> <out_dir> [step]
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk


def main(dicom_dir, out_dir, step=4):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(dicom_dir, "*.DCM")) + glob.glob(os.path.join(dicom_dir, "*.dcm"))

    # Sort slices by position along the scan axis (headers only, no pixels)
    headers = [(f, pydicom.dcmread(f, stop_before_pixels=True)) for f in files]
    headers.sort(key=lambda h: float(h[1].ImagePositionPatient[2]) if "ImagePositionPatient" in h[1]
                 else int(h[1].InstanceNumber))
    files = [f for f, _ in headers]

    h = headers[0][1]
    slope = float(getattr(h, "RescaleSlope", 1))
    intercept = float(getattr(h, "RescaleIntercept", 0))
    px = [float(v) for v in h.PixelSpacing]
    dz = float(getattr(h, "SliceThickness", px[0]) or px[0])
    print(f"slices: {len(files)}  matrix: {h.Rows}x{h.Columns}  bits: {h.BitsStored}")
    print(f"pixel spacing: {px} mm  slice thickness: {dz} mm")
    print(f"modality: {getattr(h, 'Modality', '?')}  manufacturer: {getattr(h, 'Manufacturer', '?')}  "
          f"model: {getattr(h, 'ManufacturerModelName', '?')}")
    print(f"rescale: value * {slope} + {intercept}  units: {getattr(h, 'RescaleType', 'unspecified')}")
    print(f"full volume would be {len(files) * h.Rows * h.Columns * 2 / 1e9:.1f} GB as int16")

    # Downsampled volume
    vol = np.stack([pydicom.dcmread(f).pixel_array[::step, ::step] for f in files[::step]]).astype(np.float32)
    vol = vol * slope + intercept
    print(f"loaded downsampled volume {vol.shape} (z,y,x), {vol.nbytes / 1e6:.0f} MB, "
          f"range {vol.min():.0f}..{vol.max():.0f}")

    lo, hi = np.percentile(vol, [1, 99.8])
    z, y, x = (s // 2 for s in vol.shape)
    views = [("axial", vol[z]), ("coronal", vol[:, y, :]), ("sagittal", vol[:, :, x]),
             ("axial MIP", vol.max(axis=0)), ("coronal MIP", vol.max(axis=1)), ("sagittal MIP", vol.max(axis=2))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (name, im) in zip(axes.flat, views):
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi if "MIP" not in name else vol.max(), origin="upper")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(dicom_dir.rstrip('/'))}  (every {step}th voxel)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "views.png"), dpi=90)

    # One slice at full resolution
    mid = pydicom.dcmread(files[len(files) // 2]).pixel_array * slope + intercept
    plt.figure(figsize=(10, 10))
    plt.imshow(mid, cmap="gray", vmin=lo, vmax=hi)
    plt.title(f"full-res axial slice {len(files) // 2}  ({px[0] * 1000:.0f} um pixels)")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "axial_fullres.png"), dpi=100, bbox_inches="tight")

    plt.figure(figsize=(8, 4))
    plt.hist(vol.ravel(), bins=256, log=True)
    plt.title("intensity histogram (downsampled volume)")
    plt.savefig(os.path.join(out_dir, "histogram.png"), dpi=90, bbox_inches="tight")

    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((px[1] * step, px[0] * step, dz * step))
    sitk.WriteImage(img, os.path.join(out_dir, f"volume_ds{step}.nii.gz"))
    print(f"saved views.png, axial_fullres.png, histogram.png, volume_ds{step}.nii.gz -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
