import pandas as pd
import requests
import os
import re
import time
import signal
import multiprocessing as mp
from alive_progress import alive_it, alive_bar
from country_list import countries_for_language
from typing import Dict
from pathlib import Path


_e = mp.Event()

def init_worker(e):
    """
    Tells worker processes to ignore SIGINT (Ctrl+C).
    This ensures they don't die instantly and let the main process handle the shutdown.
    Also passes mp.event() to worker processes.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _e
    _e = e


def sanitize_filename(name):
    """Removes invalid characters from strings to ensure safe file paths."""
    return re.sub(r'[\\/*?:"<>|]', "", str(name)).strip()


def download_sound(row, output_dir: str, country: str, countries: Dict[str, str] | None):
    """Worker function to download a single audio file."""

    if _e.is_set():
        return ""

    if countries is None:
        countries = dict(countries_for_language("en"))

    url = row.get("Identifier")
    scientific_name = row.get("scientificName", "Unknown_Taxon")
    country = row.get("country", countries[country])
    # media_type = row.get("type")
    record_id = row.get("CoreId")

    # Get just the internal ID without the source initials ("@XC" in this case)
    record_id = record_id[:-3]

    if pd.isna(url) or not str(url).startswith("http"):
    # if pd.isna(url) or pd.isna(media_type) or not str(url).startswith("http"):
        return f"Skipped (Invalid URL): {url}"

    # Determine file extension based on type
    # if media_type == "Sound":
    #     ext = "mp3"
    # elif media_type == "StillImage":
    #     ext = "png"
    ext = "mp3"
    # else:
    #     return f"Skipped (Invalid media type): {type}"  # Skip unhandled media types

    # Clean strings for safe file creation
    safe_name = sanitize_filename(scientific_name).replace(" ", "_")
    safe_country = sanitize_filename(country).replace(" ", "_")

    # Construct base filename: {scientificName}_{country}.mp3
    # To prevent overwriting files with the exact same name, we append the unique ID
    # Format: {scientificName}_{country}_{id}.mp3
    # filename = f"{safe_name}_{safe_country}_{record_id}.mp3"
    # filename = f"{safe_name}_{safe_country}"

    # filename = f"{safe_name}_US_{record_id}.mp3"
    # if ext == "mp3":
    #     foldername = Path(f"{output_dir}/{safe_name.replace("_", " ")}/Sound")
    # else:
    #     foldername = Path(f"{output_dir}/{safe_name.replace("_", " ")}/Spectrogram")
    foldername = Path(f"{output_dir}/{safe_name.replace("_", " ")}/Sound")

    foldername.mkdir(parents=True, exist_ok=True)
    filename = f"{record_id}.{ext}"
    filepath = os.path.join(foldername, filename)
    if Path(filepath).exists():
        return f"Skipped (File Exists): {filepath}"

    max_retries = 5
    retry_delay = 30
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=15)

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=2 ** 20):  # Chunk sizes of 10MB
                    f.write(chunk)
            return f"Success: {filename}"

        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                if attempt > 0:
                    print(
                        f"ConnectionError for {filename}. Retrying in {retry_delay} seconds... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Skipping {filename}: Failed after {max_retries} connection attempts.")
                return f"Failed after {max_retries} connection attempts."

        except Exception as e:
            return f"Failed: {filename} - Error: {str(e)}"
    return ""


def pre(row):
    # Create a directory to store the downloaded mp3 files
    country = "US"
    output_dir = f"{country}/downloaded_sounds"
    os.makedirs(output_dir, exist_ok=True)
    countries = dict(countries_for_language("en"))

    download_sound(row, output_dir, country, countries)


def main():
    # 1. Load the data
    print("Loading CSV files...")
    df_multi = pd.read_csv("Multimedia.txt", low_memory=False)
    df_occ = pd.read_csv("Occurrence.txt", low_memory=False)

    # 2. Merge the datasets on CoreId == id
    print("Merging data...")
    df_merged = pd.merge(df_multi, df_occ, left_on="CoreId", right_on="id", how="inner")

    # Memory management
    del df_multi, df_occ

    # 3. Filter for country == 'United States' // and type == 'Sound'
    df_filtered = df_merged[
        (df_merged["type"].str.lower() == "sound") &
        (df_merged["country"].str.lower() == "united states")
        ]
    # df_filtered = df_merged[df_merged["country"].str.lower() == "united states"]

    # Memory management
    del df_merged

    # Convert dataframe rows to a list of dictionaries for multiprocessing
    tasks = df_filtered.to_dict('records')

    # Memory management
    del df_filtered

    print(f"Found {len(tasks)} files to download. Starting pool...")

    # 4. Execute using multiprocessing
    e = mp.Event()
    try:
        # Pool.map automatically handles distributing a unique set of tasks to each core
        with alive_bar(len(tasks), bar="classic") as bar, mp.Pool(processes=14, initializer=init_worker,
                                                                  initargs=(e,)) as pool:
            for _ in pool.imap_unordered(pre, tasks):
                bar()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Waiting for child processes to finish...")
        e.set()

    else:
        print("\nTerminating processes for unknown reason...")
        e.set()

    # Print a quick summary of operations
    # successes = sum(1 for r in results if r.startswith("Success"))
    # print(f"\nDownload Summary: {successes} successful, {len(tasks) - successes} failed/skipped.")


if __name__ == '__main__':
    main()
