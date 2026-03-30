#!/usr/bin/env python3
"""
Fetch HDB resale flat prices from data.gov.sg (dataset: registration date from Jan-2017).
See: https://data.gov.sg — search "resale flat prices" for the official dataset page.
"""

import json
import sys
import time
import zipfile
from pathlib import Path

import requests

# Official dataset ID (HDB resale, Jan-2017 onwards)
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

API_OPEN_BASE = "https://api-open.data.gov.sg"
API_METADATA_BASE = "https://api-production.data.gov.sg"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATE_FILE = DATA_DIR / ".hdb_resale_version.json"
OUTPUT_CSV = DATA_DIR / "resale.csv"


def get_dataset_metadata(dataset_id: str) -> dict | None:
    url = f"{API_METADATA_BASE}/v2/public/api/datasets/{dataset_id}/metadata"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out = r.json()
        if out.get("code") in (0, 200) and "data" in out:
            return out["data"]
        return None
    except requests.RequestException as e:
        print(f"Metadata request failed: {e}", file=sys.stderr)
        return None


def load_last_version() -> str | None:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        return data.get("lastUpdatedAt")
    except (json.JSONDecodeError, OSError):
        return None


def save_version(last_updated_at: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"lastUpdatedAt": last_updated_at, "dataset_id": DATASET_ID}, indent=2)
    )


def initiate_download(dataset_id: str, retry_on_429: bool = True) -> tuple[bool, str | None]:
    url = f"{API_OPEN_BASE}/v1/public/api/datasets/{dataset_id}/initiate-download"
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                if retry_on_429 and attempt == 0:
                    print("Rate limited (429). Waiting 65s then retrying...", file=sys.stderr)
                    time.sleep(65)
                    continue
                print("Rate limited (429). Wait a minute and retry.", file=sys.stderr)
                return False, None
            r.raise_for_status()
            data = r.json()
            if data.get("code") not in (0, 200, 201):
                print(f"Initiate download response: {data}", file=sys.stderr)
                return False, None
            d = data.get("data") or {}
            return True, d.get("url")
        except requests.RequestException as e:
            print(f"Initiate download failed: {e}", file=sys.stderr)
            return False, None
    return False, None


def poll_download_url(dataset_id: str, max_wait_sec: int = 120, poll_interval: int = 3) -> str | None:
    url = f"{API_OPEN_BASE}/v1/public/api/datasets/{dataset_id}/poll-download"
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            data = r.json()
            if data.get("code") not in (200, 201):
                time.sleep(poll_interval)
                continue
            d = data.get("data") or {}
            status = (d.get("status") or "").upper()
            if status == "READY" and d.get("url"):
                return d["url"]
            time.sleep(poll_interval)
        except requests.RequestException as e:
            print(f"Poll failed: {e}", file=sys.stderr)
            time.sleep(poll_interval)
    return None


def download_file(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return False


def is_zip_file(path: Path) -> bool:
    with open(path, "rb") as f:
        header = f.read(4)
    return header[:2] == b"PK"


def extract_csv_from_zip(zip_path: Path, out_csv: Path) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            print("No CSV found in zip.", file=sys.stderr)
            return False
        csv_name = csv_names[0]
        for n in csv_names:
            if "resale" in n.lower():
                csv_name = n
                break
        with zf.open(csv_name) as src, open(out_csv, "wb") as dst:
            dst.write(src.read())
    return True


def run_pipeline(force_download: bool = False) -> bool:
    metadata = get_dataset_metadata(DATASET_ID)
    if not metadata:
        print("Could not fetch dataset metadata.", file=sys.stderr)
        return False

    remote_version = metadata.get("lastUpdatedAt")
    if not remote_version:
        print("Metadata has no lastUpdatedAt.", file=sys.stderr)
        return False

    last_version = load_last_version()
    if not force_download and last_version == remote_version:
        print(f"Dataset unchanged (lastUpdatedAt={remote_version}). Skipping download.")
        return True

    print(f"New version detected: {remote_version} (previous: {last_version or 'none'}). Downloading...")

    time.sleep(2)
    ok, download_url = initiate_download(DATASET_ID)
    if not ok:
        return False
    if not download_url:
        download_url = poll_download_url(DATASET_ID)
    if not download_url:
        print("Poll-download did not return a URL in time.", file=sys.stderr)
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "resale_download.tmp"

    if not download_file(download_url, raw_path):
        return False

    if is_zip_file(raw_path):
        if not extract_csv_from_zip(raw_path, OUTPUT_CSV):
            return False
        try:
            raw_path.unlink()
        except OSError:
            pass
    else:
        raw_path.rename(OUTPUT_CSV)

    save_version(remote_version)
    print(f"Updated {OUTPUT_CSV} (lastUpdatedAt={remote_version}).")
    return True


def main() -> int:
    force = "--force" in sys.argv or "-f" in sys.argv
    ok = run_pipeline(force_download=force)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
