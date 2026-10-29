"""
Download remaining VizWiz-Priv training images into data/cache/train_images/.
Strictly preserves existing val_images and never touches test split.
Logs failures to results/download_failures.csv.
"""

import os
import io
import time
import json
import csv
import struct
import zlib
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

REMOTE_URL = "https://vizwiz.cs.colorado.edu/dataset_downloads/VizWiz-Priv-v1.0/HoleMean_Images.zip"
TRAIN_ANN_PATH = Path("data/raw/Annotations/Annotations/train.json")
VAL_ANN_PATH = Path("data/raw/Annotations/Annotations/val.json")
TEST_ANN_PATH = Path("data/raw/Annotations/Annotations/test.json")

TRAIN_CACHE_DIR = Path("data/cache/train_images")
VAL_CACHE_DIR = Path("data/cache/val_images")
RESULTS_DIR = Path("results")
FAILURES_LOG = RESULTS_DIR / "download_failures.csv"

MAX_WORKERS = 12
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0


class RemoteZipReader:
    """Read specific member files from a remote ZIP archive via HTTP range requests."""

    def __init__(self, url: str, timeout: int = 20):
        self.url = url
        self.timeout = timeout
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.length = int(resp.headers["Content-Length"])
        self.entries = {}
        self._load_central_directory()

    def _read_range(self, start: int, size: int) -> bytes:
        end = start + size - 1
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def _load_central_directory(self):
        tail_size = min(65536, self.length)
        tail_data = self._read_range(self.length - tail_size, tail_size)
        eocd_idx = tail_data.rfind(b"\x50\x4b\x05\x06")
        if eocd_idx == -1:
            raise ValueError(f"End of Central Directory not found in {self.url}")

        eocd = tail_data[eocd_idx : eocd_idx + 22]
        _, _, _, total_entries, cd_size, cd_offset = struct.unpack("<HHHHII", eocd[4:20])
        cd_data = self._read_range(cd_offset, cd_size)

        pos = 0
        while pos < len(cd_data):
            if cd_data[pos : pos + 4] != b"\x50\x4b\x01\x02":
                break
            compression = struct.unpack("<H", cd_data[pos + 10 : pos + 12])[0]
            comp_size = struct.unpack("<I", cd_data[pos + 20 : pos + 24])[0]
            uncomp_size = struct.unpack("<I", cd_data[pos + 24 : pos + 28])[0]
            fn_len = struct.unpack("<H", cd_data[pos + 28 : pos + 30])[0]
            extra_len = struct.unpack("<H", cd_data[pos + 30 : pos + 32])[0]
            comment_len = struct.unpack("<H", cd_data[pos + 32 : pos + 34])[0]
            local_offset = struct.unpack("<I", cd_data[pos + 42 : pos + 46])[0]
            fn = cd_data[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="ignore")

            base_fn = os.path.basename(fn)
            if base_fn:
                self.entries[base_fn] = {
                    "full_path": fn,
                    "compression": compression,
                    "comp_size": comp_size,
                    "uncomp_size": uncomp_size,
                    "local_offset": local_offset,
                }
            pos += 46 + fn_len + extra_len + comment_len

    def get_file_bytes(self, filename: str) -> bytes:
        base_fn = os.path.basename(filename)
        if base_fn not in self.entries:
            raise KeyError(f"File {filename} not in remote index")

        info = self.entries[base_fn]
        header_data = self._read_range(info["local_offset"], 30)
        local_fn_len = struct.unpack("<H", header_data[26:28])[0]
        local_extra_len = struct.unpack("<H", header_data[28:30])[0]
        data_offset = info["local_offset"] + 30 + local_fn_len + local_extra_len
        raw_data = self._read_range(data_offset, info["comp_size"])

        if info["compression"] == 0:
            return raw_data
        elif info["compression"] == 8:
            return zlib.decompress(raw_data, -15)
        else:
            raise ValueError(f"Unsupported compression {info['compression']}")


def verify_image_bytes(data: bytes) -> bool:
    """Verifies that bytes can be fully parsed and loaded as a valid RGB image."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def verify_image_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def download_single(reader: RemoteZipReader, filename: str, out_path: Path):
    """Download a single image with exponential backoff retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_bytes = reader.get_file_bytes(filename)
            if not verify_image_bytes(raw_bytes):
                raise ValueError("Downloaded data failed image verification")
            # Write to a temp file and rename atomically
            temp_path = out_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(raw_bytes)
            temp_path.replace(out_path)
            return True, None
        except Exception as e:
            if attempt == MAX_RETRIES:
                return False, f"Attempt {attempt}: {str(e)}"
            time.sleep(INITIAL_BACKOFF * (2 ** (attempt - 1)))


def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting image download & verification...", flush=True)
    TRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load train annotations
    with open(TRAIN_ANN_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_filenames = [x["image"] for x in train_data]
    total_train = len(train_filenames)
    print(f"Total train images in official annotation: {total_train}", flush=True)

    # 2. Check existing verified images
    existing_verified = 0
    to_download = []
    for fn in train_filenames:
        target = TRAIN_CACHE_DIR / fn
        if target.exists() and verify_image_file(target):
            existing_verified += 1
        else:
            to_download.append(fn)

    print(f"Already cached and verified: {existing_verified}", flush=True)
    print(f"Remaining to download: {len(to_download)}", flush=True)

    if not to_download:
        print("All train images are already downloaded and verified.", flush=True)
        return

    # 3. Initialize RemoteZipReader
    print(f"Loading central directory from {REMOTE_URL}...", flush=True)
    reader = RemoteZipReader(REMOTE_URL)
    print(f"Remote central directory loaded ({len(reader.entries)} files in remote archive).", flush=True)

    # 4. Filter images present in remote zip
    missing_from_remote = []
    valid_targets = []
    for fn in to_download:
        if fn in reader.entries:
            valid_targets.append(fn)
        else:
            missing_from_remote.append(fn)

    failures = []
    for fn in missing_from_remote:
        failures.append({"filename": fn, "reason": "Not present in remote ZIP archive"})

    print(f"Download targets in archive: {len(valid_targets)}, missing in archive: {len(missing_from_remote)}", flush=True)

    # 5. Multithreaded download
    completed = 0
    success_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fn = {
            executor.submit(download_single, reader, fn, TRAIN_CACHE_DIR / fn): fn
            for fn in valid_targets
        }

        for future in as_completed(future_to_fn):
            fn = future_to_fn[future]
            completed += 1
            try:
                ok, err = future.result()
                if ok:
                    success_count += 1
                else:
                    failures.append({"filename": fn, "reason": err or "Failed after retries"})
            except Exception as e:
                failures.append({"filename": fn, "reason": f"Unhandled exception: {str(e)}"})

            if completed % 100 == 0 or completed == len(valid_targets):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"[{time.strftime('%H:%M:%S')}] Progress: {completed}/{len(valid_targets)} "
                    f"({completed/len(valid_targets)*100:.1f}%) | Success: {success_count} | "
                    f"Failures: {len(failures)} | Speed: {rate:.1f} img/s",
                    flush=True
                )

    # 6. Log failures to CSV
    with open(FAILURES_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reason"])
        writer.writeheader()
        for fail in failures:
            writer.writerow(fail)

    final_verified = sum(1 for fn in train_filenames if verify_image_file(TRAIN_CACHE_DIR / fn))
    print("\n--- Final Download Summary ---", flush=True)
    print(f"Total official train samples: {total_train}", flush=True)
    print(f"Successfully cached & verified on disk: {final_verified}", flush=True)
    print(f"Logged failures: {len(failures)} (saved to {FAILURES_LOG})", flush=True)
    print(f"Total time elapsed: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
