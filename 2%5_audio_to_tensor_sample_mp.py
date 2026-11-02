import multiprocessing as mp
import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import csv
import warnings
from collections import defaultdict
from pydub import AudioSegment
from pathlib import Path
from torchcodec.decoders import AudioDecoder
from alive_progress import alive_bar


warnings.filterwarnings("ignore")


def process_mp3(file_path):
    """
    Worker function to apply workers to files.
    """
    try:
        unsup = "Sample rate unsupported (i.e. too low)."
        scientific_name = Path(file_path).parts[-3]

        return True, scientific_name

    except Exception as e:
        return False, f"Error processing {file_path}: {e}"


def main():
    base_dir = Path("US")
    unsup = "Sample rate unsupported (i.e. too low)."

    if not base_dir.exists():
        print(f"Directory '{base_dir}' does not exist.")
        return

    # 1. Gather all unique file paths
    print(f"Scanning '{base_dir}' for MP3 files...")
    file_paths = [str(path) for path in base_dir.rglob("*.mp3")]

    total_files = len(file_paths)
    if total_files == 0:
        print("No MP3 files found.")
        return

    print(f"Found {total_files} files.")

    # 2. Configure multiprocessing pool
    num_cores = mp.cpu_count()
    num_cores = 14
    print(f"Starting pool with {num_cores} cores...")

    # chunksize groups the unique paths into batches for each core.
    # A chunksize > 1 reduces the overhead of handing out files one-by-one.
    chunk_size = max(1, total_files // (num_cores * 4))
    chunk_size = 1

    success_count = 0
    fail_count = 0
    numpy_dict = {}

    # 3. Process using imap_unordered
    with alive_bar(len(file_paths), bar="classic") as bar, mp.Pool(processes=num_cores) as pool:
        # imap_unordered takes the unique list of paths and distributes them
        # without duplication to the available workers.
        results = pool.imap_unordered(process_mp3, file_paths, chunksize=chunk_size)

        # Consume the iterator as files finish processing
        for status, message in results:
            if status:
                success_count += 1
                numpy_dict[message] = 0
            else:
                fail_count += 1
                # Print errors immediately on a new line
                if message != unsup:
                    print(f"\n[FAIL] {message}\t\t|\t{fail_count = } ")

            # Print a dynamic progress indicator
            bar()

    padded_values = []
    ordered_keys = []
    for b, s in numpy_dict.items():
        ordered_keys.append(b)
    ordered_keys = [k for k in numpy_dict.keys()]
    del numpy_dict

    x = np.array(padded_values, dtype=float)
    print("Saving Class Mapping...")
    headers = ["ClassName", "ClassID"]
    with open("class_mapping_us.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for id, cl in enumerate(ordered_keys):
            writer.writerow([cl, id])
    x = torch.from_numpy(x)
    filename = "dataset/audio_dataset.pt"
    torch.save(x, filename)
    print(f"\n\nComplete! Successfully processed: {success_count} | Failed: {fail_count}")


if __name__ == '__main__':
    main()


# Expands the lists so each tensor has its own row with the matching index
# df_expanded = df.explode("Tensors")
# print(df_expanded)