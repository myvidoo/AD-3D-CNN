#!/usr/bin/env python3
# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/download_targets.py  (translated from the authors' archive path)
# [Paper correspondence] MIRIAD target-list generation
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [14, 16]  (line numbers refer to the original Chinese version)

# Targeted MIRIAD downloader — ONLY the first test (baseline _1_MR_1) of each subject.
# Targets (experiment IDs resolved from metadata_experiments.csv):
#   miriad_188_1_MR_1 -> atropychallenge_E01737
#   miriad_189_1_MR_1 -> atropychallenge_E01166
#   miriad_190_1_MR_1 -> atropychallenge_E01203
# Resilient: probes the server; when the file-servlet recovers it downloads the 3
# zips (15s apart), extracts NIfTI, and stops. If the cookie expires it stops and
# asks for a fresh JSESSIONID. Resume-safe (skips already-present valid zips).
# Usage:  python -u download_targets.py <JSESSIONID>
import sys, os, zipfile, time, urllib.parse
import urllib.request, urllib.error

BASE = "http://miriad.drc.ion.ucl.ac.uk/atrophychallenge"
PROJ = "MIRIAD"
OUT = r"<LOCAL_HOME>/<tool_dir>/<session>/miriad"
NII = os.path.join(OUT, "nii")
os.makedirs(NII, exist_ok=True)

COOKIE = sys.argv[1] if len(sys.argv) > 1 else open(os.path.join(OUT, "cookie.txt")).read().strip()
open(os.path.join(OUT, "cookie.txt"), "w").write(COOKIE)

# (experiment ID, human label) — the 3 baseline sessions
TARGETS = [
    ("atropychallenge_E01737", "miriad_188_1_MR_1"),
    ("atropychallenge_E01166", "miriad_189_1_MR_1"),
    ("atropychallenge_E01203", "miriad_190_1_MR_1"),
]

GAP = 15
PROBE_WAIT = 600
MAX_PROBES = 1000

opener = urllib.request.build_opener()
opener.addheaders.append(("Cookie", f"JSESSIONID={COOKIE}"))

def fetch(url, timeout=300):
    with opener.open(url, timeout=timeout) as r:
        return r.read()

def classify(d):
    if d[:2] == b"PK":
        return "zip"
    h = d[:500].lower()
    if b"<?xml" in h[:20]:
        return "xml"          # authenticated, but file-servlet returns metadata fallback
    if b"<html" in h or b"login" in h or b"j_spring" in h:
        return "login"        # cookie expired -> redirected to login page
    return "other"

# startup cleanup: move aside any corrupt (non-zip) *.zip
bad_dir = os.path.join(NII, "_bad_xml")
for f in os.listdir(NII):
    p = os.path.join(NII, f)
    if f.endswith(".zip") and os.path.isfile(p) and not zipfile.is_zipfile(p):
        try:
            os.makedirs(bad_dir, exist_ok=True)
            os.rename(p, os.path.join(bad_dir, f))
            print(f"[cleanup] moved aside corrupt: {f}", flush=True)
        except Exception as e:
            print(f"[cleanup] could not move {f}: {e}", flush=True)

print(f"[start] targets={len(TARGETS)} cookie={COOKIE[:8]}... gap={GAP}s", flush=True)

def probe():
    url = (f"{BASE}/data/archive/projects/{PROJ}/experiments/"
           f"atropychallenge_E01737/scans/ALL/files?format=zip")
    try:
        return classify(fetch(url, timeout=120)), None
    except Exception as e:
        return "err", str(e)

def download_all():
    ok = fail = 0
    for eid, label in TARGETS:
        dest = os.path.join(NII, f"{eid}.zip")
        if os.path.exists(dest) and os.path.getsize(dest) > 100000 and zipfile.is_zipfile(dest):
            print(f"[skip] {label} already present", flush=True)
            ok += 1
            continue
        url = (f"{BASE}/data/archive/projects/{PROJ}/experiments/"
               f"{urllib.parse.quote(eid)}/scans/ALL/files?format=zip")
        done = False
        for t in range(3):
            try:
                d = fetch(url, timeout=300)
                k = classify(d)
                if k == "zip":
                    with open(dest, "wb") as fh:
                        fh.write(d)
                    done = True
                    break
                elif k == "login":
                    return "login"
                else:
                    print(f"[ {label}] server file-servlet down (kind={k}); backoff", flush=True)
                    time.sleep(60)
            except Exception as e:
                print(f"[ {label}] try {t+1} err: {e}", flush=True)
                time.sleep(30)
        if not done:
            print(f"[GAVEUP] {label}", flush=True)
            fail += 1
            continue
        try:
            with zipfile.ZipFile(dest) as z:
                for n in z.namelist():
                    if n.endswith(".nii.gz") or n.endswith(".nii"):
                        z.extract(n, NII)
            print(f"[ok] {label} -> NIfTI extracted", flush=True)
            ok += 1
        except Exception as e:
            print(f"[extract fail] {label}: {e}", flush=True)
            fail += 1
        time.sleep(GAP)
    return ("done", ok, fail)

result = None
for p in range(MAX_PROBES):
    kind, err = probe()
    print(f"[probe {p+1}/{MAX_PROBES}] {kind} {err or ''}", flush=True)
    if kind == "zip":
        print("[probe] file-servlet OK -> downloading 3 baseline sessions", flush=True)
        res = download_all()
        if res == "login":
            print("[stop] cookie expired -> provide fresh JSESSIONID", flush=True)
            result = "cookie-expired"
            break
        if isinstance(res, tuple) and res[0] == "done":
            print(f"[done] baseline sessions downloaded: ok={res[1]} fail={res[2]}", flush=True)
            result = "done"
            break
        print("[probe] server broke mid-run -> re-entering wait", flush=True)
    elif kind == "login":
        print("[stop] cookie EXPIRED -> re-login in browser, copy JSESSIONID, re-run", flush=True)
        result = "cookie-expired"
        break
    else:
        print(f"[probe] server file-servlet unavailable; waiting {PROBE_WAIT}s", flush=True)
        time.sleep(PROBE_WAIT)

if result is None:
    print("[end] gave up after max probes (server still down). Re-run later or contact MIRIAD admin.", flush=True)
else:
    print(f"[end] result={result}", flush=True)
