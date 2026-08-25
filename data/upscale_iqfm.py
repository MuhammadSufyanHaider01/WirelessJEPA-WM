# upscale_iqfm.py
import argparse
import os
import h5py
import numpy as np

def upscale_h5(input_path, output_path, factor=64, axis=1, chunk_size=1024, compression=None):
    """
    Upscale the 'iq_data' dataset along a chosen axis using nearest-neighbor
    (np.repeat), writing a new HDF5 file with the same labels copied over.
    Expected original shape: (N, A, C, T). Default axis=1 (antenna).
    """
    assert os.path.exists(input_path), f"Input not found: {input_path}"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(input_path, "r") as fin:
        if "iq_data" not in fin:
            raise ValueError(f"'iq_data' not found in {input_path}. Keys: {list(fin.keys())}")

        iq = fin["iq_data"]
        orig_shape = iq.shape  # (N, A, C, T)
        dtype = iq.dtype

        if axis < 0:
            axis = axis % len(orig_shape)
        if not (0 <= axis < len(orig_shape)):
            raise ValueError(f"axis must be in [0,{len(orig_shape)-1}]")

        # Compute output shape
        out_shape = list(orig_shape)
        out_shape[axis] *= factor
        out_shape = tuple(out_shape)

        with h5py.File(output_path, "w") as fout:
            # Create upscaled iq_data dataset
            dset = fout.create_dataset(
                "iq_data",
                shape=out_shape,
                dtype=dtype,
                compression=compression,  # e.g., "lzf" or "gzip"
                chunks=True
            )

            # Copy all other top-level datasets verbatim if present
            for key in fin.keys():
                if key == "iq_data":
                    continue
                src = fin[key]
                if isinstance(src, h5py.Dataset):
                    fout.create_dataset(key, data=src[...], dtype=src.dtype, compression=compression, chunks=True)
                else:
                    # Shallow-copy groups if any (rare in your case)
                    fin.copy(src, fout, name=key)

            # Chunked nearest-neighbor upscaling along `axis`
            N = orig_shape[0]
            start = 0
            while start < N:
                end = min(start + chunk_size, N)
                # Read a chunk: shape (B, A, C, T)
                batch = iq[start:end]  # numpy array
                # Repeat along axis
                up_batch = np.repeat(batch, repeats=factor, axis=axis)
                dset[start:end] = up_batch
                start = end

            # Optional: store some metadata
            dset.attrs["upscale_axis"] = axis
            dset.attrs["upscale_factor"] = factor
            dset.attrs["source_file"] = os.path.basename(input_path)

    print(f"✓ Wrote upscaled file: {output_path} (shape: {out_shape})")


def main():
    parser = argparse.ArgumentParser(description="Upscale IQFM HDF5 along antenna axis to avoid runtime repeat_interleave.")
    parser.add_argument("--train", type=str, default="./data/iqfm-train.h5", help="Path to input train HDF5")
    parser.add_argument("--val", type=str, default="./data/iqfm-val.h5", help="Path to input val HDF5")
    parser.add_argument("--outdir", type=str, default="./data", help="Output directory")
    parser.add_argument("--factor", type=int, default=64, help="Upscale factor (e.g., 64 for 4→256 antennas)")
    parser.add_argument("--axis", type=int, default=1, help="Axis to upscale (default 1 = antenna in (N,A,C,T))")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Number of samples per write chunk")
    parser.add_argument("--compression", type=str, default=None, help='HDF5 compression: None, "lzf", or "gzip"')
    args = parser.parse_args()

    train_out = os.path.join(args.outdir, "iqfm-upscaled-train.h5")
    val_out   = os.path.join(args.outdir, "iqfm-upscaled-val.h5")

    # upscale_h5(args.train, train_out, factor=args.factor, axis=args.axis,
    #            chunk_size=args.chunk_size, compression=args.compression)
    upscale_h5(args.val,   val_out,   factor=args.factor, axis=args.axis,
               chunk_size=args.chunk_size, compression=args.compression)

if __name__ == "__main__":
    main()
