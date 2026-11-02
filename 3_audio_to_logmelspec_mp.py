import os

import librosa
import skimage.io
import numpy as np
import torch
import pandas as pd
import multiprocessing as mp
from pathlib import Path
import torch.nn.functional as F
import torchaudio
from torchcodec.decoders import AudioDecoder
import torchaudio.transforms as T
from torchvision.transforms import v2
from alive_progress import alive_bar


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def process_mp3(file_path):
    """
    Worker function to apply workers to files.
    """
    try:
        scientific_name = Path(file_path).parts[-3]

        stem = Path(file_path).stem
        dec = AudioDecoder(file_path, num_channels=1)

        if dec.metadata.duration_seconds > 25:
            y = dec.get_samples_played_in_range(5, 20)
            y = y.data

            # y, orig_sr = librosa.load(
            #     file_path,
            #     sr=None,
            #     mono=True,
            #     offset=5,
            #     duration=15
            # )
        elif 15 < dec.metadata.duration_seconds <= 25:
            y = dec.get_samples_played_in_range(0, 15)
            y = y.data

            # y, orig_sr = librosa.load(
            #     file_path,
            #     sr=None,
            #     mono=True,
            #     duration=10
            # )
        else:
            y = dec.get_samples_played_in_range()

            # y, orig_sr = librosa.load(
            #     file_path,
            #     sr=None,
            #     mono=True
            # )
            needed_to_pad = (dec.metadata.sample_rate * 15) - y.data.shape[-1]
            left_pad = needed_to_pad // 2
            right_pad = needed_to_pad - left_pad
            y = F.pad(y.data, (left_pad, right_pad, 0, 0), mode="constant", value=0)

        # step2 - converting audio np array to spectrogram
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=dec.metadata.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        mel_spec = mel_spectrogram_transform(y.data)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        mel_spec_db = to_db(mel_spec)
        mel_spec_db = mel_spec_db.detach()
        resize_tool = v2.Resize((64,256))
        img = resize_tool(mel_spec_db)
        img = (img - img.min()) / (img.max() - img.min())

        print(scientific_name)

        return True, (img, scientific_name)

    except Exception as e:
        return False, (f"Error processing {file_path}: {e}", file_path)


def main():
    base_dir = Path("US")
    dataset_dir = "dataset/"

    if not base_dir.exists():
        print(f"Directory '{base_dir}' does not exist.")
        return

    # 1. Gather all unique file paths
    print(f"Scanning '{base_dir}' for MP3 files...")
    file_paths = [str(path) for path in base_dir.rglob("*.mp3")]
    # file_paths = file_paths[:200]

    total_files = len(file_paths)
    if total_files == 0:
        print("No MP3 files found.")
        return

    print(f"Found {total_files} files.")

    # 2. Configure multiprocessing pool
    num_cores = mp.cpu_count()
    num_cores = 2
    print(f"Starting pool with {num_cores} cores...")

    # chunksize groups the unique paths into batches for each core.
    # A chunksize > 1 reduces the overhead of handing out files one-by-one.
    chunk_size = max(1, total_files // (num_cores * 4))
    chunk_size = 1

    success_count = 0
    fail_count = 0

    class_mappings = pd.read_csv("class_mapping_us.csv", index_col="ClassName")
    features = []
    labels = []

    # 3. Process using imap_unordered
    with alive_bar(len(file_paths), bar="classic") as bar, mp.Pool(processes=num_cores) as pool:
        # imap_unordered takes the unique list of paths and distributes them
        # without duplication to the available workers.
        results = pool.imap_unordered(process_mp3, file_paths, chunksize=chunk_size)

        # Consume the iterator as files finish processing
        for status, message in results:
            if status:
                success_count += 1
                img_tensor, class_name = message
                class_label = class_mappings.loc[class_name, class_mappings.columns[0]]
                features.append(img_tensor)
                labels.append(torch.tensor(class_label))
            else:
                fail_count += 1
                # Print errors immediately on a new line
                print(f"\n[FAIL] {message[0]}\t\t|\t{fail_count = } ")
                os.remove(message[1])

            # Print a dynamic progress indicator
            bar()
    features = torch.stack(features)
    labels = torch.stack(labels)
    torch.save(features, f"{dataset_dir}features.pt")
    torch.save(labels, f"{dataset_dir}labels.pt")


if __name__ == '__main__':
    main()
