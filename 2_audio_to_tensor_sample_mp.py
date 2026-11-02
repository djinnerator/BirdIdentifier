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
        target_sr = 44100
        y, orig_sr = librosa.load(file_path, sr=None, mono=True)
        if orig_sr > target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        elif orig_sr < target_sr:
            return False, unsup
        y_int16 = (y * 32768).astype(np.int16)
        audio_seg = AudioSegment(data=y_int16.tobytes(), sample_width=y_int16.dtype.itemsize, frame_rate=target_sr,
                                 channels=1)
        audio_buffer = io.BytesIO()
        audio_seg.export(audio_buffer, format="mp3")
        audio_buffer.seek(0)
        enc_bytes = audio_buffer.read()
        decoder = AudioDecoder(enc_bytes)
        duration = decoder.metadata.duration_seconds
        if duration > 20:
            samples = decoder.get_samples_played_in_range(
                10,
                20
            )
        elif 10 < duration <= 20:
            samples = decoder.get_samples_played_in_range(
                decoder.metadata.begin_stream_seconds,
                decoder.metadata.begin_stream_seconds + 10
            )
        else:
            samples = decoder.get_all_samples()
            needed_to_pad = (samples.sample_rate * 10) - samples.data.size(-1)
            samples.data = F.pad(samples.data, (1, needed_to_pad - 1))

        return True, (scientific_name, samples.data.numpy())

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
    file_paths = file_paths[:200]

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
    numpy_dict = defaultdict(list)

    # 3. Process using imap_unordered
    with alive_bar(len(file_paths), bar="classic") as bar, mp.Pool(processes=num_cores) as pool:
        # imap_unordered takes the unique list of paths and distributes them
        # without duplication to the available workers.
        results = pool.imap_unordered(process_mp3, file_paths, chunksize=chunk_size)

        # Consume the iterator as files finish processing
        for status, message in results:
            if status:
                success_count += 1
                numpy_dict[message[0]].append(message[1])
            else:
                fail_count += 1
                # Print errors immediately on a new line
                if message != unsup:
                    print(f"\n[FAIL] {message}\t\t|\t{fail_count = } ")

            # Print a dynamic progress indicator
            bar()

    audio_shape = (1, 441000)
    longest_len = max(len(s) for s in numpy_dict.values())
    empty_audio = np.zeros(audio_shape, dtype=float)
    padded_values = []
    ordered_keys = []
    for b, s in numpy_dict.items():
        ordered_keys.append(b)
        num_missing = longest_len - len(s)
        padded_samples = list(s) + [empty_audio] * num_missing
        padded_values.append(padded_samples)
    del numpy_dict

    x = np.array(padded_values, dtype=float)
    print("Saving Class Mapping...")
    headers = ["ClassID", "ClassName"]
    with open("class_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for id, cl in enumerate(ordered_keys):
            writer.writerow([id, cl])
    x = torch.from_numpy(x)
    filename = "dataset/audio_dataset.pt"
    torch.save(x, filename)
    print(f"\n\nComplete! Successfully processed: {success_count} | Failed: {fail_count}")


if __name__ == '__main__':
    main()


# Expands the lists so each tensor has its own row with the matching index
# df_expanded = df.explode("Tensors")
# print(df_expanded)