# -*- coding: utf-8 -*-
"""Correctness verification ① for the structural measurements: a two-hypothesis blind test of the Harvard-Oxford atlas label mapping.

[why this verification is needed]
The NIfTI image label values of the Harvard-Oxford atlas are shifted by 1 overall relative to
the <label index> in the XML:
  · index=0 in HarvardOxford-Subcortical.xml is "Left Cerebral White Matter", and the XML
    **has no background entry**;
  · whereas label value 0 in the NIfTI file is background.
Hence **image label value = XML index + 1**. If you wrongly assume "image value = XML index" and
take the hippocampus (value 8/18), you actually get the **brainstem** — a pitfall that silently
produces wrong results.

[verification method]
For every image label value (1..21) of the subcortical atlas, compute the voxel count + MNI
centroid and compare it against 21 known anatomical centroids (tolerance 30 mm), scoring each
of the two hypotheses:
    Hypothesis A: value k = XML[k]        (no shift)
    Hypothesis B: value k = XML[k−1]      (shift by 1)
Whichever scores more hits is the true mapping.

[expected result]
  Hypothesis A = 15/21, Hypothesis B = 21/21  → establishing "image value = XML index + 1"
  The decisive case: image value 8 = 38,610 mm³ with centroid (0.6,−31.0,−34.2), which can only
              be the brainstem (the hippocampus is about 11,000 mm³ with centroid near (±25,−22,−14))

Run:
    python baselines/verify_atlas_identity.py
"""
import os
import io
import re
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# expected anatomical centroid of each structure (MNI mm)
EXPECT = {
    "Left Cerebral White Matter": (-25, -17, 18), "Left Cerebral Cortex": (-38, -20, 12),
    "Left Lateral Ventrical": (-8, 2, 10), "Left Thalamus": (-11, -19, 9),
    "Left Caudate": (-13, 9, 9), "Left Putamen": (-25, 1, 0),
    "Left Pallidum": (-20, -5, -1), "Brain-Stem": (0, -30, -35),
    "Left Hippocampus": (-25, -22, -14), "Left Amygdala": (-23, -5, -18),
    "Left Accumbens": (-10, 12, -8),
    "Right Cerebral White Matter": (25, -17, 18), "Right Cerebral Cortex": (38, -20, 12),
    "Right Lateral Ventricle": (8, 2, 10), "Right Thalamus": (11, -19, 9),
    "Right Caudate": (13, 9, 9), "Right Putamen": (25, 1, 0),
    "Right Pallidum": (20, -5, -1), "Right Hippocampus": (25, -22, -14),
    "Right Amygdala": (23, -5, -18), "Right Accumbens": (10, 12, -8),
}
TOL_MM = 30.0


def main():
    cfg = load_config()
    sub_p = resolve_path(cfg, "atlas.harvard_oxford_sub")

    # the XML sits one level above the image directory (.../data/atlases/HarvardOxford/ and .../data/atlases/)
    xml_p = None
    for cand in (
        os.path.join(os.path.dirname(sub_p), "HarvardOxford-Subcortical.xml"),
        os.path.join(os.path.dirname(os.path.dirname(sub_p)), "HarvardOxford-Subcortical.xml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sub_p))),
                     "HarvardOxford-Subcortical.xml"),
    ):
        if os.path.exists(cand):
            xml_p = cand
            break
    if xml_p is None:
        print("⚠ Atlas XML (HarvardOxford-Subcortical.xml) not found.\n"
              "  This verification needs the XML metadata of the FSL atlas; if only the NIfTI\n"
              "  images are installed, this script can be skipped — extract_structural_panel_s9.py\n"
              "  has already hard-coded the label mapping established by this verification\n"
              "  (image value = XML index + 1).\n"
              f"  The locations tried are based on atlas.harvard_oxford_sub in config.yaml: {sub_p}")
        return

    txt = open(xml_p, encoding="ISO-8859-1", errors="replace").read()
    labs = {int(i): n.strip() for i, n in
            re.findall(r'<label index="(\d+)"[^>]*>(.*?)</label>', txt)}
    print(f"Atlas XML: {xml_p} ({len(labs)} labels)\n")

    img = nib.load(sub_p)
    d = np.asanyarray(img.dataobj)
    aff = img.affine

    print(f"{'value':>4} {'n(mm³)':>8} {'centroid MNI':>22}   "
          f"{'Hypothesis A: value k=XML[k]':<24} {'Hypothesis B: value k=XML[k-1]':<24}")
    print("-" * 104)

    score_a = score_b = 0
    for v in range(1, 22):
        m = d == v
        n = int(m.sum())
        if n == 0:
            continue
        c = aff[:3, :3] @ np.array(np.nonzero(m)).mean(1) + aff[:3, 3]

        def err(name):
            e = EXPECT.get(name)
            return float(np.linalg.norm(c - np.array(e))) if e else 9e9

        name_a, name_b = labs.get(v, "?"), labs.get(v - 1, "?")
        win_a, win_b = err(name_a) < TOL_MM, err(name_b) < TOL_MM
        score_a += win_a
        score_b += win_b
        print(f"{v:>4} {n:>8}  ({c[0]:6.1f},{c[1]:6.1f},{c[2]:6.1f})   "
              f"{name_a[:22]:<22}{'hit' if win_a else '×':<2}  "
              f"{name_b[:22]:<22}{'hit' if win_b else '×':<2}")

    print("-" * 104)
    print(f"Anatomical hits: Hypothesis A (value k=XML[k]) = {score_a}/21   "
          f"Hypothesis B (value k=XML[k-1]) = {score_b}/21")
    verdict = ("the image label values are shifted by +1 overall relative to the XML indices" if score_b > score_a
               else "the image label values correspond directly to the XML indices")
    print(f"\nConclusion: {verdict}")

    if score_b > score_a:
        print("\n★ The ROI values are therefore: hippocampus 9/19, amygdala 10/20, lateral ventricle 3/14, brainstem 8, "
              "parahippocampal gyrus anterior 34 / posterior 35")


if __name__ == "__main__":
    main()
