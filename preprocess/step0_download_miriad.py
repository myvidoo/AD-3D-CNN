#!/usr/bin/env python3
# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/download_miriad_v5.py  (translated from the authors' archive path)
# [Paper correspondence] MIRIAD download (resume-safe session-level zip downloader)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [20, 22]  (line numbers refer to the original Chinese version)

# MIRIAD resilient watcher + downloader (v5)
# ===========================================================================
# Why this exists: the MIRIAD XNAT file-servlet intermittently returns the
# experiment's MRSession XML instead of the actual scan zip (a server-side
# issue that also causes the website "Download all" -> "An error has occurred").
# This script:
#   1) PROBES whether the file-servlet has recovered (1 cheap request).
#   2) When it returns a real zip (magic 'PK'), downloads all 708 mrSessionData
#      sessions as per-session zips, 15s apart, extracting NIfTI.
#   3) If it sees a LOGIN PAGE -> cookie expired -> stops, asks for fresh cookie.
#   4) If it sees the XML fallback -> server down -> waits 10 min, retries
#      (up to 6 times ~1h), then gives up.
# Resume-safe: skips sessions already present as valid zips.
# Usage:  python -u download_miriad_v5.py <JSESSIONID>
# ===========================================================================
import sys, os, io, glob, zipfile, csv, time
import urllib.request, urllib.error

BASE = "http://miriad.drc.ion.ucl.ac.uk/atrophychallenge"
PROJ = "MIRIAD"
OUT = r"<LOCAL_HOME>/<tool_dir>/<session>/miriad"
NII = os.path.join(OUT, "nii")
os.makedirs(NII, exist_ok=True)

COOKIE = sys.argv[1] if len(sys.argv) > 1 else open(os.path.join(OUT, "cookie.txt")).read().strip()
open(os.path.join(OUT, "cookie.txt"), "w").write(COOKIE)  # remember for resume

GAP = 15            # polite gap between session downloads (s)
PROBE_WAIT = 600    # wait between recovery probes when server is down (s)
MAX_PROBES = 1000   # ~1 week of 10-min probes; real stop only on cookie expiry or recovery

opener = urllib.request.build_opener()
opener.addheaders.append(("Cookie", f"JSESSIONID={COOKIE}"))

def fetch(url, timeout=300):
    with opener.open(url, timeout=timeout) as r:
        return r.read()

def classify(data):
    if data[:2] == b"PK":
        return "zip"
    head = data[:500].lower()
    if b"<?xml" in head[:20]:
        return "xml"          # authenticated, but file-servlet returns metadata fallback
    if b"<html" in head or b"login" in head or b"j_spring" in head:
        return "login"        # cookie expired -> redirected to login page
    return "other"

# --- startup: move corrupt (non-zip) *.zip aside so nii/ stays clean ---
bad_dir = os.path.join(NII, "_bad_xml")
for f in glob.glob(os.path.join(NII, "*.zip")):
    if not zipfile.is_zipfile(f):
        try:
            os.makedirs(bad_dir, exist_ok=True)
            os.rename(f, os.path.join(bad_dir, os.path.basename(f)))
            print(f"[cleanup] moved aside corrupt: {os.path.basename(f)}", flush=True)
        except Exception as e:
            print(f"[cleanup] could not move {os.path.basename(f)}: {e}", flush=True)

# --- load experiment list from local CSV (avoids extra requests) ---
with open(os.path.join(OUT, "metadata_experiments.csv"), encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r.get("xsiType") == "xnat:mrSessionData"]
ids = [r.get("ID") or r.get("label") for r in rows if (r.get("ID") or r.get("label"))]
print(f"[start] cookie={COOKIE[:8]}... experiments={len(ids)} gap={GAP}s", flush=True)

def probe():
    url = f"{BASE}/data/archive/projects/{PROJ}/experiments/atropychallenge_E01045/scans/ALL/files?format=zip"
    try:
        return classify(fetch(url, timeout=120)), None
    except Exception as e:
        return "err", str(e)

def download_all():
    ok = fail = 0
    for i, eid in enumerate(ids, 1):
        dest = os.path.join(NII, f"{eid}.zip")
        if os.path.exists(dest) and os.path.getsize(dest) > 100000 and zipfile.is_zipfile(dest):
            ok += 1
            continue
        url = (f"{BASE}/data/archive/projects/{PROJ}/experiments/"
               f"{urllib.parse.quote(eid)}/scans/ALL/files?format=zip")
        done = False
        for t in range(3):
            try:
                d = fetch(url, timeout=300)
                kind = classify(d)
                if kind == "zip":
                    with open(dest, "wb") as fh:
                        fh.write(d)
                    done = True
                    break
                elif kind == "login":
                    return "login"
                else:
                    print(f"[exp {i}] server file-servlet down (kind={kind}); backoff", flush=True)
                    time.sleep(60)
            except Exception as e:
                print(f"[exp {i}] try {t+1} err: {e}", flush=True)
                time.sleep(30)
        if not done:
            print(f"[exp {i}] GAVE UP {eid}", flush=True)
            fail += 1
            continue
        try:
            with zipfile.ZipFile(dest) as z:
                for n in z.namelist():
                    if n.endswith(".nii.gz") or n.endswith(".nii"):
                        z.extract(n, NII)
            ok += 1
        except Exception as e:
            print(f"[exp {i}] extract fail {eid}: {e}", flush=True)
            fail += 1
        if i % 10 == 0 or i == len(ids):
            print(f"[progress] {i}/{len(ids)} ok={ok} fail={fail}", flush=True)
        time.sleep(GAP)
    return ("done", ok, fail)

# --- main loop ---
result = None
for p in range(MAX_PROBES):
    kind, err = probe()
    print(f"[probe {p+1}/{MAX_PROBES}] {kind} {err or ''}", flush=True)
    if kind == "zip":
        print("[probe] file-servlet OK -> starting full download", flush=True)
        res = download_all()
        if res == "login":
            print("[stop] cookie expired -> provide fresh JSESSIONID", flush=True)
            result = "cookie-expired"
            break
        if isinstance(res, tuple) and res[0] == "done":
            print(f"[done] downloaded NIfTI for {res[1]}/{len(ids)} (fail={res[2]})", flush=True)
            result = "done"
            break
        # res == 'serverdown' -> broke mid-run, fall through to waiting
        print("[probe] server broke mid-run -> re-entering wait", flush=True)
    elif kind == "login":
        print("[stop] cookie EXPIRED -> re-login in browser, copy JSESSIONID, re-run with it", flush=True)
        result = "cookie-expired"
        break
    else:
        print(f"[probe] server file-servlet unavailable; waiting {PROBE_WAIT}s", flush=True)
        time.sleep(PROBE_WAIT)

if result is None:
    print("[end] gave up after max probes (server still down). Re-run later or contact MIRIAD admin.", flush=True)
else:
    print(f"[end] result={result}", flush=True)
