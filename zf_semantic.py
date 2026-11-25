#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# MV2, Munim, Inc. Made with AI agents working in MV2's name.
"""Pre-registered re-test of the 'semantic magnet' comparison in Elie et al. 2025,
Science 389:1210 ("Categorical and semantic perception of the meaning of call types in
zebra finches"), on the authors' own code and data (Zenodo 15795568, v3.0 = GitHub
commit 4a3c6ef, CC BY 4.0).

See PREREG.md. Subcommands:
  fetch --data-dir D --out-dir O   download + md5-check the authors' archive
  run   --data-dir D --out-dir O   three notebook arms + scale-free tests + grade

Only numpy is imported at module level so the unit tests run with numpy alone.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import sys
import time
import traceback
import types
import urllib.request
import zipfile

import numpy as np

RECORD = "15795568"
ZIP_URL = ("https://zenodo.org/api/records/15795568/files/"
           "theunissenlab/zebra-finch-semantic-category-3.0.zip/content")
ZIP_MD5 = "eb041f1c9928a7feadb206ef955168e6"
ZIP_SIZE = 99341119
ZIP_NAME = "zebra-finch-semantic-category-3.0.zip"
NB_NAME = "BEHAV-Figure 4.ipynb"
UA = {"User-Agent": "mv2-research-job/1.0 (open research; Apache-2.0)"}

# The one-letter typo: the shifted behavioural matrix is assigned to a new name.
TYPO_OLD = "confMatdist = confMatDist - minval"
TYPO_NEW = "confMatDist = confMatDist - minval"
# The acoustic matrix does receive its shift; the 'noshift' arm removes it.
ACSHIFT_OLD = "confMatDistAcoust = confMatAcoustLOR - minval"
ACSHIFT_NEW = "confMatDistAcoust = confMatAcoustLOR - 0.0*minval"
# Record every bootstrap pair of distance matrices (read-only side effect).
REC_OLD = "behMDSBoot = mds.fit_transform(confMatDist)"
REC_NEW = ("behMDSBoot = mds.fit_transform(confMatDist); _BOOT_REC.append(("
           "np.array(confMatDist, dtype=float).copy(), "
           "np.array(confMatDistAcoust, dtype=float).copy()))")

ARMS = {
    "asis": [(REC_OLD, REC_NEW)],
    "fixed": [(TYPO_OLD, TYPO_NEW), (REC_OLD, REC_NEW)],
    "noshift": [(ACSHIFT_OLD, ACSHIFT_NEW), (REC_OLD, REC_NEW)],
}

CALL_LIST_DEFAULT = ["So", "DC", "LT", "Te", "Tu", "Th", "Ag", "Di", "Wh", "Ne", "Be"]
NAME_GRP_DEFAULT = ["Ag", "DC", "Ne", "Te", "Wh", "Di", "So", "Th", "Tu", "Be", "LT"]
SEM_DEFAULT = {"Contact": ["LT", "DC", "Te"], "Alarm": ["Tu", "Th"], "Fight": ["Di", "Ag"],
               "Bond": ["Wh", "Ne"], "Display": ["So"], "Hunger": ["Be"]}
PUBLISHED = {"beh_q": 26.37, "ac_q": 10.89, "count": 992,
             "beh_q_noso": 21.27, "ac_q_noso": 9.68, "count_noso": 979, "nboot": 1000}
SNAP_NAMES = ["confMatTot", "confMatAvg", "confMatCountAcoustic", "callList", "name_grp",
              "indINname_grp", "semCluster", "semClusterNoSo"]
SEED = 20261125
N_CAL = 200


# ----------------------------------------------------------------------------- helpers

def rankdata(a):
    """Average ranks starting at 1 (ties share the mean rank)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def sym_offdiag(m):
    m = np.array(m, dtype=float)
    s = (m + m.T) / 2.0
    np.fill_diagonal(s, 0.0)
    return s


def upper_vec(s):
    s = np.asarray(s, dtype=float)
    iu = np.triu_indices(s.shape[0], 1)
    return s[iu]


def group_ids(labels, sem):
    """Group index per label; labels in no group get their own singleton id."""
    lab2g = {}
    for gi, (_, members) in enumerate(sorted(sem.items())):
        for m in members:
            lab2g[m] = gi
    out, nxt = [], len(sem)
    for lab in labels:
        if lab in lab2g:
            out.append(lab2g[lab])
        else:
            out.append(nxt)
            nxt += 1
    return out


def same_vec(ids):
    ids = np.asarray(ids)
    iu0, iu1 = np.triu_indices(ids.size, 1)
    return (ids[iu0] == ids[iu1]).astype(float)


def block_sizes(ids):
    _, counts = np.unique(np.asarray(ids), return_counts=True)
    return tuple(sorted(int(c) for c in counts))


def set_partitions(n, sizes):
    """All partitions of range(n) into unlabeled blocks with the given multiset of sizes.
    Returns an int array (P, n) of block ids. Each partition appears exactly once."""
    out = []

    def rec(elems, sizes_left, assign, next_id):
        if not elems:
            out.append(list(assign))
            return
        first, rest = elems[0], elems[1:]
        for sz in sorted(set(sizes_left)):
            left_sizes = list(sizes_left)
            left_sizes.remove(sz)
            for comb in itertools.combinations(rest, sz - 1):
                assign[first] = next_id
                for e in comb:
                    assign[e] = next_id
                remaining = [e for e in rest if e not in comb]
                rec(remaining, left_sizes, assign, next_id + 1)

    if sum(sizes) != n:
        raise ValueError("sizes do not sum to n")
    rec(list(range(n)), list(sizes), [0] * n, 0)
    return np.array(out, dtype=int)


def partition_same_matrix(n, sizes):
    a = set_partitions(n, sizes)
    iu0, iu1 = np.triu_indices(n, 1)
    return (a[:, iu0] == a[:, iu1]).astype(float)


def partial_T(y, x, S):
    """Semantic coefficient (sign flipped: positive = same-group pairs closer) in the
    regression y ~ 1 + x + s, for every row s of S. y and x are pair vectors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    X = np.column_stack([np.ones_like(x), x])
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    yr = y - H @ y
    Sr = S - S @ H
    num = Sr @ yr
    den = np.sum(Sr * Sr, axis=1)
    T = np.full(S.shape[0], np.nan)
    ok = den > 1e-12
    T[ok] = -num[ok] / den[ok]
    return T


def exact_p(T_all, T_obs):
    T_all = np.asarray(T_all, dtype=float)
    T_all = T_all[np.isfinite(T_all)]
    return float(np.sum(T_all >= T_obs - 1e-9)) / float(T_all.size)


def auc_within(dvec, s):
    """P(a same-group pair is closer than a cross-group pair); ties count half."""
    dvec = np.asarray(dvec, dtype=float)
    s = np.asarray(s, dtype=float)
    w = dvec[s == 1]
    a = dvec[s == 0]
    if w.size == 0 or a.size == 0:
        return float("nan")
    lt = (w[:, None] < a[None, :]).sum()
    eq = (w[:, None] == a[None, :]).sum()
    return float(lt + 0.5 * eq) / float(w.size * a.size)


def auc_all(dvec, S):
    """auc_within for every row of S (all rows must have the same number of 1s)."""
    S = np.atleast_2d(np.asarray(S, dtype=float))
    r = rankdata(dvec)
    nw = int(round(S[0].sum()))
    na = S.shape[1] - nw
    ra = (1.0 - S) @ r
    return (ra - na * (na + 1) / 2.0) / float(nw * na)


def acoustic_lor(counts):
    """The authors' acoustic log2 odds-ratio (off-diagonal) from LDA confusion counts:
    zero counts -> 0.5, row-normalise, L_ij = log2(((1-p_ij)/p_ij) / ((1-p_ii)/p_ii))."""
    c = np.array(counts, dtype=float)
    if c.ndim == 3:
        c = c.sum(axis=0)
    c = np.where(c == 0, 0.5, c)
    p = c / c.sum(axis=1, keepdims=True)
    d = np.diag(p)
    odd_ii = (1.0 - d) / d
    lor = np.log2(((1.0 - p) / p) / odd_ii[:, None])
    np.fill_diagonal(lor, 0.0)
    return lor


def reorder_index(name_grp, call_list):
    return [list(name_grp).index(c) for c in call_list]


def clean_source(src):
    """Drop IPython magics / shell lines (only unindented '%name' or '!name' lines)."""
    keep = []
    for line in src.split("\n"):
        if re.match(r"^%{1,2}[A-Za-z]", line) or re.match(r"^![A-Za-z]", line):
            keep.append("pass  # removed: " + line.replace("\\", "/")[:80])
            continue
        keep.append(line.replace("get_ipython()", "_FakeIPython()"))
    return "\n".join(keep)


def apply_patches(src, patches, counts):
    for old, new in patches:
        k = src.count(old)
        counts[old] = counts.get(old, 0) + k
        src = src.replace(old, new)
    return src


def headline(loader_ok, reproduced, p_fixed, p_partial):
    if not loader_ok or p_fixed is None or p_partial is None:
        return "LOADER_STOP"
    if not reproduced:
        return "NOT_REPRODUCED"
    q_ok = p_fixed >= 0.95
    e_ok = p_partial <= 0.05
    if q_ok and e_ok:
        return "SURVIVES_BOTH"
    if (not q_ok) and e_ok:
        return "TYPO_DEPENDENT_SEMANTIC_HOLDS"
    if (not q_ok) and (not e_ok):
        return "TYPO_DEPENDENT_NOT_SUPPORTED"
    return "Q_SURVIVES_PARTIAL_FAILS"


def brier(preds):
    """preds: list of (prior, outcome_bool or None). Mean Brier over graded ones."""
    g = [(p, o) for p, o in preds if o is not None]
    if not g:
        return None
    return float(np.mean([(p - (1.0 if o else 0.0)) ** 2 for p, o in g]))


# ----------------------------------------------------------------------------- fetch

def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(a):
    dest = os.path.join(a.data_dir, "zenodo-" + RECORD)
    os.makedirs(dest, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(dest, ZIP_NAME)
    man = {"record": RECORD, "url": ZIP_URL, "expected_md5": ZIP_MD5, "expected_size": ZIP_SIZE}
    try:
        req = urllib.request.Request("https://zenodo.org/api/records/" + RECORD, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            meta = json.loads(r.read().decode("utf-8"))
        md = meta.get("metadata", {})
        lic = md.get("license") or {}
        man["api"] = {"title": md.get("title"), "version": md.get("version"),
                      "license": lic.get("id") if isinstance(lic, dict) else lic,
                      "files": [{"key": f.get("key"), "size": f.get("size"),
                                 "checksum": f.get("checksum")} for f in meta.get("files", [])]}
    except Exception as e:  # metadata is informative only
        man["api_error"] = repr(e)[:300]
    if not (os.path.exists(path) and md5_of(path) == ZIP_MD5):
        req = urllib.request.Request(ZIP_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r, open(path, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
    man["md5"] = md5_of(path)
    man["size"] = os.path.getsize(path)
    man["ok"] = (man["md5"] == ZIP_MD5 and man["size"] == ZIP_SIZE)
    with open(os.path.join(a.out_dir, "download_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("download ok" if man["ok"] else "DOWNLOAD PROBLEM", man["md5"], man["size"])
    return 0 if man["ok"] else 1


# ----------------------------------------------------------------------------- notebook arms

class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _StubObj()


class _StubObj(object):
    def __call__(self, *a, **k):
        return _StubObj()

    def __getattr__(self, name):
        return _StubObj()


class _FakeIPython(object):
    def __getattr__(self, name):
        return lambda *a, **k: None


def _install_stub(name):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        nm = ".".join(parts[:i])
        if nm not in sys.modules:
            sys.modules[nm] = _Stub(nm)


def run_notebook(nb_path, patches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    plt.show = lambda *a, **k: None
    plt.savefig = lambda *a, **k: None
    Figure.savefig = lambda self, *a, **k: None

    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    ns = {"__name__": "__main__", "display": print, "_BOOT_REC": [],
          "_FakeIPython": _FakeIPython}
    counts, snaps, log, stubs = {}, {}, [], []
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(nb_path))
    try:
        for i, c in enumerate(cells):
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            src = apply_patches(clean_source(src), patches, counts)
            t0 = time.time()
            status, out, err = "ok", "", ""
            for attempt in range(6):
                buf, ebuf = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(ebuf):
                        exec(compile(src, "<cell %d>" % i, "exec"), ns)
                    status = "ok"
                except ModuleNotFoundError as e:
                    if e.name and attempt < 5:
                        _install_stub(e.name)
                        stubs.append(e.name)
                        continue
                    status = "error: " + repr(e)[:300]
                    err = traceback.format_exc()[-3000:]
                except (Exception, SystemExit) as e:
                    status = "error: " + repr(e)[:300]
                    err = traceback.format_exc()[-3000:]
                out = buf.getvalue()
                err = (ebuf.getvalue()[-2000:] + err)[-5000:]
                break
            try:
                plt.close("all")
            except Exception:
                pass
            log.append({"cell": i, "status": status, "seconds": round(time.time() - t0, 2),
                        "first_line": src.split("\n")[0][:120],
                        "stdout": out[-15000:], "stderr": err})
            for nm in SNAP_NAMES:
                if nm in ns and nm not in snaps:
                    try:
                        snaps[nm] = copy.deepcopy(ns[nm])
                    except Exception:
                        snaps[nm] = ns[nm]
    finally:
        os.chdir(old_cwd)
    return ns, log, snaps, counts, stubs


def _find_notebook(root):
    for dp, _, files in os.walk(root):
        if NB_NAME in files and os.path.basename(dp) == "pythonNotebooks":
            return os.path.join(dp, NB_NAME)
    return None


def _arr(ns, name):
    v = ns.get(name)
    if v is None:
        return None
    try:
        return np.asarray(v, dtype=float)
    except Exception:
        return None


def _printed(log, pattern):
    rx = re.compile(pattern)
    hits = []
    for e in log:
        for line in e["stdout"].split("\n"):
            if rx.search(line):
                hits.append(line.strip())
    return hits


def arm_summary(ns, log, counts, stubs):
    s = {"patch_counts": counts, "stubs": sorted(set(stubs)),
         "cells": len(log), "cells_error": [e["cell"] for e in log if e["status"] != "ok"]}
    for key, bname, aname in [("song", "behQBoot", "acQBoot"),
                              ("nosong", "behQBootNoSo", "acQBootNoSo")]:
        b, ac = _arr(ns, bname), _arr(ns, aname)
        if b is None or ac is None or b.shape != ac.shape or b.size == 0:
            s[key] = None
            continue
        s[key] = {"n": int(b.size), "beh_q_mean": float(np.mean(b)),
                  "ac_q_mean": float(np.mean(ac)),
                  "beh_q_2sd": float(2 * np.std(b)), "ac_q_2sd": float(2 * np.std(ac)),
                  "count_beh_gt_ac": int(np.sum(b > ac)),
                  "p_beh_gt_ac": float(np.mean(b > ac))}
    mac = _arr(ns, "macBoot")
    s["disparity_mean"] = None if mac is None else float(np.mean(mac))
    s["printed"] = _printed(log, r"Q = |Percent times|disparity|PCC|p value")[:60]
    s["n_boot_records"] = len(ns.get("_BOOT_REC", []))
    return s


# ----------------------------------------------------------------------------- analysis

def our_q(D, ids, seed=0):
    from sklearn.manifold import MDS
    D = np.asarray(D, dtype=float)
    X = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
            n_init=10).fit_transform(D)
    ids = np.asarray(ids)
    W, A = [], []
    for g in np.unique(ids):
        m = ids == g
        if m.sum() < 2:
            continue
        c = X[m].mean(axis=0)
        W.append(np.mean(np.sum((X[m] - c) ** 2, axis=1)))
        A.append(np.mean(np.sum((X[~m] - c) ** 2, axis=1)))
    return float(np.mean(A) / np.mean(W))


def scale_free_tests(Bd, Ad, ids, S_all, rng):
    """Bd, Ad: symmetric distance matrices in the same label order."""
    vb, va = upper_vec(Bd), upper_vec(Ad)
    s = same_vec(ids)
    y, x = rankdata(vb), rankdata(va)
    T_all = partial_T(y, x, S_all)
    T_obs = float(partial_T(y, x, s[None, :])[0])
    auc_b_all, auc_a_all = auc_all(vb, S_all), auc_all(va, S_all)
    res = {"n_labels": len(ids), "n_pairs": int(vb.size), "n_same_pairs": int(s.sum()),
           "n_partitions": int(S_all.shape[0]),
           "partial_T": T_obs, "partial_p": exact_p(T_all, T_obs),
           "auc_beh": auc_within(vb, s), "auc_ac": auc_within(va, s),
           "spearman_beh_ac": float(np.corrcoef(y, x)[0, 1])}
    res["auc_beh_p"] = exact_p(auc_b_all, res["auc_beh"])
    res["auc_ac_p"] = exact_p(auc_a_all, res["auc_ac"])
    # calibration: rank-scale worlds with no semantic term, and with one of size sigma
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    sigma = float(np.std(y - yhat, ddof=2))
    null_p, pow_p = [], []
    for _ in range(N_CAL):
        y0 = yhat + rng.normal(0.0, sigma, size=y.size)
        null_p.append(exact_p(partial_T(y0, x, S_all), float(partial_T(y0, x, s[None, :])[0])))
        y1 = y0 - sigma * s
        pow_p.append(exact_p(partial_T(y1, x, S_all), float(partial_T(y1, x, s[None, :])[0])))
    res["calibration"] = {"worlds": N_CAL, "sigma_rank": sigma,
                          "null_fpr_05": float(np.mean(np.array(null_p) <= 0.05)),
                          "power_05_delta_sigma": float(np.mean(np.array(pow_p) <= 0.05))}
    return res


def boot_scale_free(records, idx, ids, keep):
    s = same_vec(ids)
    ab, aa, ts = [], [], []
    for bd, ad in records:
        bd = sym_offdiag(bd)[np.ix_(keep, keep)]
        ad = sym_offdiag(ad)[np.ix_(idx, idx)][np.ix_(keep, keep)]
        vb, va = upper_vec(bd), upper_vec(ad)
        ab.append(auc_within(vb, s))
        aa.append(auc_within(va, s))
        ts.append(float(partial_T(rankdata(vb), rankdata(va), s[None, :])[0]))
    ab, aa, ts = np.array(ab), np.array(aa), np.array(ts)
    if ab.size == 0:
        return None
    return {"n": int(ab.size), "auc_beh_median": float(np.median(ab)),
            "auc_ac_median": float(np.median(aa)),
            "p_auc_beh_gt_ac": float(np.mean(ab > aa)), "p_auc_tie": float(np.mean(ab == aa)),
            "partial_T_median": float(np.median(ts)), "p_partial_T_gt0": float(np.mean(ts > 0))}


def q_demo(Ac, ids):
    off = upper_vec(Ac)
    sd = float(np.std(off))
    base = Ac - (off.min() - 0.1 * sd)
    np.fill_diagonal(base, 0.0)
    out = {}
    variants = {"base": base}
    for k in (1, 2, 4):
        v = base + k * sd
        np.fill_diagonal(v, 0.0)
        variants["plus_%dsd" % k] = v
    variants["square"] = base ** 2
    variants["cube"] = base ** 3
    e = np.exp(base / sd)
    np.fill_diagonal(e, 0.0)
    variants["exp"] = e
    for k, v in variants.items():
        try:
            out[k] = our_q(v, ids)
        except Exception as ex:
            out[k] = "error " + repr(ex)[:120]
    return out


def cmd_run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    zpath = os.path.join(a.data_dir, "zenodo-" + RECORD, ZIP_NAME)
    load = {"zip": zpath, "exists": os.path.exists(zpath)}
    load["md5_ok"] = load["exists"] and md5_of(zpath) == ZIP_MD5
    arms, snaps0, recs = {}, None, None
    if load["md5_ok"]:
        for arm, patches in ARMS.items():
            for mname in list(sys.modules):
                mfile = getattr(sys.modules[mname], "__file__", None) or ""
                if str(mfile).startswith("/tmp/zf-"):
                    del sys.modules[mname]
            root = os.path.join("/tmp", "zf-" + arm)
            shutil.rmtree(root, ignore_errors=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(root)
            nb = _find_notebook(root)
            if nb is None:
                arms[arm] = {"error": "notebook not found"}
                continue
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(nb)), "results"),
                        exist_ok=True)
            t0 = time.time()
            ns, log, snaps, counts, stubs = run_notebook(nb, patches)
            summ = arm_summary(ns, log, counts, stubs)
            summ["seconds"] = round(time.time() - t0, 1)
            arms[arm] = summ
            with open(os.path.join(a.out_dir, "arm_%s_log.json" % arm), "w",
                      encoding="utf-8") as fh:
                json.dump(log, fh, indent=1)
            if arm == "asis":
                snaps0 = snaps
                recs = list(ns.get("_BOOT_REC", []))
            del ns
            shutil.rmtree(root, ignore_errors=True)
            print("arm", arm, json.dumps({k: summ.get(k) for k in ("song", "nosong",
                                                                    "patch_counts")}))
    load["arms_run"] = list(arms)

    res = {"published": PUBLISHED, "arms": arms, "load": load}
    asis = arms.get("asis") or {}
    load["asis_ok"] = bool(asis.get("song"))
    reproduced = False
    if load["asis_ok"]:
        sg = asis["song"]
        reproduced = (abs(sg["beh_q_mean"] - PUBLISHED["beh_q"]) / PUBLISHED["beh_q"] <= 0.10
                      and abs(sg["ac_q_mean"] - PUBLISHED["ac_q"]) / PUBLISHED["ac_q"] <= 0.10
                      and sg["count_beh_gt_ac"] >= 950)
    res["reproduced"] = reproduced

    # scale-free tests on the full-data matrices from the as-is run
    sf = None
    try:
        snaps0 = snaps0 or {}

        def _get(nm, default):
            v = snaps0.get(nm)
            return default if v is None else v

        call_list = [str(c) for c in list(_get("callList", CALL_LIST_DEFAULT))]
        name_grp = [str(c) for c in list(_get("name_grp", NAME_GRP_DEFAULT))]
        try:
            sem = {str(k): [str(m) for m in v]
                   for k, v in dict(_get("semCluster", SEM_DEFAULT)).items()}
        except Exception:
            sem = SEM_DEFAULT
        load["semantic_groups_source"] = "notebook" if snaps0.get("semCluster") is not None \
            else "default"
        try:
            idx = [int(i) for i in list(snaps0.get("indINname_grp"))]
            if [name_grp[i] for i in idx] != call_list:
                idx = None
        except Exception:
            idx = None
        if idx is None:
            idx = reorder_index(name_grp, call_list)
        T = np.asarray(snaps0["confMatTot"], dtype=float)
        C = np.asarray(snaps0["confMatCountAcoustic"], dtype=float)
        load["shapes"] = {"confMatTot": list(T.shape), "confMatCountAcoustic": list(C.shape)}
        Bd = sym_offdiag(-T)
        Ad = sym_offdiag(acoustic_lor(C))[np.ix_(idx, idx)]
        load["behaviour_asis_negative_offdiag"] = int(np.sum(upper_vec(Bd) < 0))
        rng = np.random.default_rng(SEED)
        sf = {"labels": call_list, "semantic_groups": sem, "idx": idx}
        for key, drop in (("with_song", None), ("without_song", "So")):
            keep = [i for i, c in enumerate(call_list) if c != drop]
            labs = [call_list[i] for i in keep]
            ids = group_ids(labs, sem)
            S_all = partition_same_matrix(len(labs), block_sizes(ids))
            sub = scale_free_tests(Bd[np.ix_(keep, keep)], Ad[np.ix_(keep, keep)], ids,
                                   S_all, rng)
            sub["boot"] = boot_scale_free(recs or [], idx, ids, keep)
            sf[key] = sub
        ids_all = group_ids(call_list, sem)
        sf["q_demo_acoustic"] = q_demo(Ad, ids_all)
        try:
            minval = float(np.min(np.diag(-T)))
            Bf = Bd - minval
            np.fill_diagonal(Bf, 0.0)
            sf["our_q_behaviour"] = {"asis_unshifted": our_q(Bd, ids_all),
                                     "shifted_as_intended": our_q(Bf, ids_all),
                                     "shift_added": -minval}
        except Exception as e2:
            sf["our_q_behaviour"] = "error " + repr(e2)[:200]
    except Exception as e:
        load["scale_free_error"] = repr(e)[:400] + " " + traceback.format_exc()[-1500:]
        sf = None
    res["scale_free"] = sf

    fx = (arms.get("fixed") or {}).get("song")
    ns_ = (arms.get("noshift") or {}).get("song")
    p_fixed = fx["p_beh_gt_ac"] if fx else None
    p_part = sf["with_song"]["partial_p"] if sf else None
    res["headline"] = headline(load["md5_ok"] and load["asis_ok"], reproduced, p_fixed, p_part)

    def g(cond):
        try:
            return bool(cond())
        except Exception:
            return None

    sg = asis.get("song") or {}
    preds = [
        ("P1", 0.80, g(lambda: load["md5_ok"] and load["asis_ok"])),
        ("P2", 0.75, g(lambda: reproduced)),
        ("P3", 0.60, g(lambda: arms["fixed"]["patch_counts"][TYPO_OLD] == 2)),
        ("P4", 0.85, g(lambda: fx["beh_q_mean"] < sg["beh_q_mean"])),
        ("P5", 0.45, g(lambda: fx["p_beh_gt_ac"] < 0.95)),
        ("P6", 0.25, g(lambda: fx["beh_q_mean"] < fx["ac_q_mean"])),
        ("P7", 0.40, g(lambda: ns_["p_beh_gt_ac"] < 0.95)),
        ("P8", 0.50, g(lambda: sf["with_song"]["partial_p"] <= 0.05)),
        ("P9", 0.45, g(lambda: sf["without_song"]["partial_p"] <= 0.05)),
        ("P10", 0.65, g(lambda: sf["with_song"]["auc_beh"] > sf["with_song"]["auc_ac"])),
        ("P11", 0.35, g(lambda: sf["with_song"]["boot"]["p_auc_beh_gt_ac"] >= 0.95)),
        ("P12", 0.85, g(lambda: sf["with_song"]["calibration"]["null_fpr_05"] <= 0.07)),
        ("P13", 0.60, g(lambda: sf["with_song"]["calibration"]["power_05_delta_sigma"] >= 0.80)),
        ("P14", 0.75, g(lambda: sf["q_demo_acoustic"]["plus_2sd"]
                        <= 0.75 * sf["q_demo_acoustic"]["base"])),
        ("P15", 0.50, g(lambda: load["behaviour_asis_negative_offdiag"] >= 1)),
    ]
    grade = {"headline": res["headline"],
             "predictions": [{"id": i, "prior": p, "outcome": o} for i, p, o in preds],
             "brier": brier([(p, o) for _, p, o in preds])}
    with open(os.path.join(a.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    with open(os.path.join(a.out_dir, "grade.json"), "w", encoding="utf-8") as fh:
        json.dump(grade, fh, indent=1)
    with open(os.path.join(a.out_dir, "load_report.json"), "w", encoding="utf-8") as fh:
        json.dump(load, fh, indent=1, default=str)
    with open(os.path.join(a.out_dir, "RESULTS-ZF.md"), "w", encoding="utf-8") as fh:
        fh.write("# zebra-finch-semantic-test results (machine-written)\n\n")
        fh.write("Headline: %s\n\n" % res["headline"])
        for arm, summ in arms.items():
            fh.write("## arm %s\n\n```\n%s\n```\n\n" % (
                arm, json.dumps({k: summ.get(k) for k in ("song", "nosong", "patch_counts",
                                                          "stubs", "cells_error",
                                                          "n_boot_records")}, indent=1)))
        fh.write("## scale-free\n\n```\n%s\n```\n\n" % json.dumps(sf, indent=1, default=str))
        fh.write("## grade\n\n```\n%s\n```\n" % json.dumps(grade, indent=1))
    print("HEADLINE", res["headline"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run"])
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out-dir", default="/out")
    a = ap.parse_args(argv)
    return cmd_fetch(a) if a.cmd == "fetch" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
