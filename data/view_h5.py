import h5py
import numpy as np

def print_h5_description(file_path):
    """
    Prints the structure of an HDF5 (.h5) file.
    Includes groups, datasets, shapes/dtypes, and number of classes.
    """
    with h5py.File(file_path, "r") as f:
        print("Keys:", list(f.keys()))

        if "iq_data" in f:
            print(f"Data shape: {f['iq_data'].shape}")
            iq_data = f["iq_data"][:]
        
        # N, A, C, T = iq_data.shape
        # iq_reshaped = iq_data.transpose(0, 1, 3, 2).reshape(-1, C)  # (N*A*T, 2)

        # mean = iq_reshaped.mean(axis=0)
        # std = iq_reshaped.std(axis=0)

        # IQ_NORMALIZE = {"mean": mean.tolist(), "std": std.tolist()}
        # print("IQ_NORMALIZE =", IQ_NORMALIZE)

        if "modulation" in f:
            mods = f["modulation"][:]
            unique_mods = np.unique(mods)
            print(f"Modulation (first 5): {mods[:5]}")
            print(f"Number of classes: {len(unique_mods)}")
            print(f"Classes: {unique_mods}")

        elif "angles" in f:
            angles = f["angles"][:]
            unique_angles = np.unique(angles, axis=0)
            print(f"Angles (first 5): {angles[:5]}")
            print(f"Number of angle pairs: {len(unique_angles)}")

# Example usage:
if __name__ == "__main__":
    # file_path = "./data/imagenet-100-train.h5"   
    # file_path = "./data/iqfm-train.h5"   
    file_path = "./data/iqfm-val-100.h5"   
    # file_path = "./data/deepbeam.h5"   
    # file_path = "./data/RF_FingerPrinting.h5"   
    # file_path = "./data/rml2016.h5"   
    print_h5_description(file_path)
