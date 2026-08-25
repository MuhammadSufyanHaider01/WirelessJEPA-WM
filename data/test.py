#!/usr/bin/env python3
import argparse, os, sys, hashlib, h5py, random, json
from typing import Dict, Any, Tuple, List

def md5_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Safe MD5 (optional, large files are streamed)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def ds_props(d: h5py.Dataset) -> Dict[str, Any]:
    """Capture key structural properties of a dataset."""
    return {
        "class": "dataset",
        "shape": tuple(d.shape),
        "dtype": str(d.dtype),
        "maxshape": tuple(d.maxshape) if d.maxshape is not None else None,
        "chunks": tuple(d.chunks) if d.chunks is not None else None,
        "compression": d.compression,
        "compression_opts": d.compression_opts,
        "shuffle": bool(d.shuffle),
        "fletcher32": bool(d.fletcher32),
        "fillvalue": d.fillvalue,
        "attrs": {k: _attr_to_jsonable(d.attrs[k]) for k in d.attrs.keys()},
    }

def grp_props(g: h5py.Group) -> Dict[str, Any]:
    return {
        "class": "group",
        "attrs": {k: _attr_to_jsonable(g.attrs[k]) for k in g.attrs.keys()},
    }

def _attr_to_jsonable(x):
    try:
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", errors="replace")
        if hasattr(x, "tolist"):
            return x.tolist()
        json.dumps(x)  # will throw if not jsonable
        return x
    except Exception:
        return str(x)

def snapshot_h5(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Walk the file and capture a flat dict of objects keyed by full path."""
    meta = {}
    items: Dict[str, Dict[str, Any]] = {}
    with h5py.File(path, "r") as f:
        meta = {
            "filename": os.path.abspath(path),
            "swmr_mode": bool(getattr(f, "swmr_mode", False)),
            "keys": list(f.keys()),
        }
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                items["/"+name] = ds_props(obj)
            elif isinstance(obj, h5py.Group):
                items["/"+name] = grp_props(obj)
        f.visititems(visitor)
    return items, meta

def diff_dicts(a: Dict[str, Any], b: Dict[str, Any], path: str, diffs: List[str]):
    """Recursive diff for simple nested dicts."""
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    for k in sorted(a_keys - b_keys):
        diffs.append(f"- only in A: {path}.{k}")
    for k in sorted(b_keys - a_keys):
        diffs.append(f"- only in B: {path}.{k}")
    for k in sorted(a_keys & b_keys):
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            diff_dicts(va, vb, f"{path}.{k}", diffs)
        else:
            if va != vb:
                diffs.append(f"- {path}.{k} differs: A={va!r}  vs  B={vb!r}")

def compare_snapshots(A: Dict[str, Dict[str, Any]], B: Dict[str, Dict[str, Any]]) -> List[str]:
    diffs: List[str] = []

    a_paths = set(A.keys())
    b_paths = set(B.keys())

    only_a = sorted(a_paths - b_paths)
    only_b = sorted(b_paths - a_paths)

    for p in only_a:
        diffs.append(f"[MISSING IN B] {p}  ({A[p]['class']})")
    for p in only_b:
        diffs.append(f"[MISSING IN A] {p}  ({B[p]['class']})")

    for p in sorted(a_paths & b_paths):
        if A[p]["class"] != B[p]["class"]:
            diffs.append(f"[TYPE DIFF] {p}: A={A[p]['class']} vs B={B[p]['class']}")
            continue
        cls = A[p]["class"]
        if cls == "dataset":
            keys = ["shape","dtype","maxshape","chunks","compression","compression_opts",
                    "shuffle","fletcher32","fillvalue"]
            for k in keys:
                if A[p].get(k) != B[p].get(k):
                    diffs.append(f"[DATASET PROP DIFF] {p}.{k}: A={A[p].get(k)} vs B={B[p].get(k)}")
            # attrs
            diff_dicts(A[p].get("attrs", {}), B[p].get("attrs", {}), f"{p}.attrs", diffs)
        elif cls == "group":
            diff_dicts(A[p].get("attrs", {}), B[p].get("attrs", {}), f"{p}.attrs", diffs)
    return diffs

def safe_sample_reads(path: str, dataset_names: List[str], n: int = 5) -> List[str]:
    """Try reading first/middle/last and random indices of listed datasets."""
    msgs: List[str] = []
    try:
        with h5py.File(path, "r") as f:
            for ds_name in dataset_names:
                if ds_name not in f:
                    msgs.append(f"[READ] {path}: dataset '{ds_name}' not found")
                    continue
                d = f[ds_name]
                if not isinstance(d, h5py.Dataset):
                    msgs.append(f"[READ] {path}: '{ds_name}' is not a dataset")
                    continue
                length = d.shape[0] if d.shape else 1
                idxs = set([0, max(0,length//2), max(0,length-1)])
                while len(idxs) < min(n, length):
                    idxs.add(random.randrange(length))
                for i in sorted(idxs):
                    try:
                        _ = d[i]
                    except Exception as e:
                        msgs.append(f"[READ ERROR] {path}:{ds_name}[{i}] -> {type(e).__name__}: {e}")
                        break
                else:
                    msgs.append(f"[READ OK] {path}:{ds_name} (sampled {len(idxs)} indices)")
    except Exception as e:
        msgs.append(f"[OPEN ERROR] {path}: {type(e).__name__}: {e}")
    return msgs

def main():
    ap = argparse.ArgumentParser(description="Compare two HDF5 files (structure, attrs, layout).")
    ap.add_argument("working")
    ap.add_argument("problem")
    ap.add_argument("--skip-md5", action="store_true", help="Skip MD5 calculation.")
    ap.add_argument("--sample-read", nargs="*", default=[], help="Dataset names to probe-read (e.g., iq_data labels).")
    ap.add_argument("--n", type=int, default=5, help="How many indices to sample-read per dataset (max).")
    args = ap.parse_args()

    A_path = os.path.abspath(args.working)
    B_path = os.path.abspath(args.problem)

    print(f"[A] working : {A_path}")
    print(f"[B] problem : {B_path}")

    if not args.skip_md5:
        try:
            print(f"[MD5] A: {md5_file(A_path)}")
            print(f"[MD5] B: {md5_file(B_path)}")
        except Exception as e:
            print(f"[MD5] error: {e}")

    A_snap, A_meta = snapshot_h5(A_path)
    B_snap, B_meta = snapshot_h5(B_path)

    print("\n[FILE META]")
    print("A:", json.dumps(A_meta, indent=2))
    print("B:", json.dumps(B_meta, indent=2))

    print("\n[STRUCTURE DIFF]")
    diffs = compare_snapshots(A_snap, B_snap)
    if not diffs:
        print("No structural differences found (groups/datasets/props/attrs match).")
    else:
        for line in diffs:
            print(line)

    if args.sample_read:
        print("\n[SAMPLE READ TESTS]")
        for msg in safe_sample_reads(A_path, args.sample_read, n=args.n):
            print(msg)
        for msg in safe_sample_reads(B_path, args.sample_read, n=args.n):
            print(msg)

    print("\nDone.")

if __name__ == "__main__":
    main()

