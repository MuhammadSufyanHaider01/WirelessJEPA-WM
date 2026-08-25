import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def sample_aoa_per_class(src_path, dst_path, per_class=100, seed=42, allow_replacement=False, block=65536):
    rng = np.random.default_rng(seed)
    src_path, dst_path = Path(src_path), Path(dst_path)

    with h5py.File(src_path, "r") as src:
        if "iq_data" not in src or "angles" not in src:
            raise ValueError("Source H5 must contain 'iq_data' and 'angles' datasets.")
        iq = src["iq_data"]
        angles = src["angles"][:]
        modulation = src["modulation"][:]

        classes, inverse = np.unique(angles, axis=0, return_inverse=True)
        n_classes = classes.shape[0]

        selected = []
        print(f"Selecting {per_class} samples per class across {n_classes} classes...")
        for cid in tqdm(range(n_classes), desc="Classes"):
            idxs = np.flatnonzero(inverse == cid)
            if idxs.size == 0:
                continue
            if idxs.size < per_class and not allow_replacement:
                print(f"[WARN] Class {classes[cid].tolist()} has only {idxs.size} samples (< {per_class}). Taking all.")
                pick = idxs
            else:
                pick = rng.choice(idxs, size=per_class, replace=allow_replacement)
            selected.append(pick)

        if not selected:
            raise ValueError("No samples selected. Check your source file and labels.")

        # Final desired order (we shuffle to mix classes)
        selected_indices = np.concatenate(selected)
        rng.shuffle(selected_indices)

        # ---- READ SAFELY: strictly increasing, no duplicates ----
        # Unique indices & sorted for h5py; also supports allow_replacement=True
        uniq_idx = np.unique(selected_indices)
        sorted_uniq_idx = np.sort(uniq_idx)  # strictly increasing

        print("Reading selected iq_data samples (sorted unique indices)...")
        # Read in blocks using sorted_uniq_idx
        out_blocks = []
        for i in tqdm(range(0, sorted_uniq_idx.size, block), desc="Reading blocks"):
            sl = sorted_uniq_idx[i:i+block]
            out_blocks.append(iq[sl])
        subset_iq_sorteduniq = np.concatenate(out_blocks, axis=0)

        # Map sorted_uniq_idx back to the desired original order with searchsorted
        # For each original index, find its position in sorted_uniq_idx
        pos_in_sorteduniq = np.searchsorted(sorted_uniq_idx, selected_indices)
        subset_iq = subset_iq_sorteduniq[pos_in_sorteduniq]
        subset_angles = angles[selected_indices]
        subset_modulation = modulation[selected_indices]

        print("Writing new HDF5 file...")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            # Streamed write with progress (optional: chunked manual write)
            d_iq = dst.create_dataset(
                "iq_data",
                shape=subset_iq.shape,
                dtype=subset_iq.dtype,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            d_ang = dst.create_dataset(
                "angles",
                shape=subset_angles.shape,
                dtype=subset_angles.dtype,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            d_mod = dst.create_dataset(
                "modulation",
                shape=subset_modulation.shape,
                dtype=subset_modulation.dtype,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )

            # Write in blocks with progress
            n = subset_iq.shape[0]
            for i in tqdm(range(0, n, block), desc="Writing blocks"):
                j = min(i + block, n)
                d_iq[i:j] = subset_iq[i:j]
                d_ang[i:j] = subset_angles[i:j]
                d_mod[i:j] = subset_modulation[i:j]

            dst.attrs["source"] = str(src_path)
            dst.attrs["stratified_by"] = "angles (rows as classes)"
            dst.attrs["per_class_requested"] = per_class
            dst.attrs["allow_replacement"] = int(allow_replacement)
            dst.attrs["seed"] = seed

        print(f"[OK] Wrote {selected_indices.size} samples to: {dst_path}")

# Example
if __name__ == "__main__":
    sample_aoa_per_class(
        src_path="./data/iqfm-val.h5",
        dst_path="./data/iqfm-val-100.h5",
        per_class=100,
        seed=42,
        allow_replacement=False,
        block=65536,  # tune for memory / speed
    )
