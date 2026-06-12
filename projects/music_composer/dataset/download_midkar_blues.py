#!/usr/bin/env python3
"""
Download all MIDI files linked from the Blues MIDIs page on midkar.com.

Usage:
  python download_midkar_blues.py
  python download_midkar_blues.py --url https://midkar.com/Blues/Blues_MIDIs.html --out midis --delay 0.5

The script skips files that already exist and saves into `./midis` (relative to this script).
"""

import argparse
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MID-Downloader/1.0; +https://github.com/)"
}


def find_mid_links(page_url):
    resp = requests.get(page_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".mid"):
            links.append(urljoin(page_url, href))
    return links


def download_file(url, out_dir, delay=0.3):
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename:
        return False, "no-filename"
    target = os.path.join(out_dir, filename)
    if os.path.exists(target):
        return False, "exists"

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = r.headers.get("content-length")
            total = int(total) if total and total.isdigit() else None
            chunk_size = 8192
            with open(target, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        time.sleep(delay)
        return True, "downloaded"
    except Exception as e:
        # If partially written file exists, remove it
        if os.path.exists(target):
            try:
                os.remove(target)
            except Exception:
                pass
        return False, str(e)


def main():
    p = argparse.ArgumentParser(description="Download all .mid files from a page (midkar Blues example)")
    p.add_argument("--url", default="https://midkar.com/Blues/Blues_MIDIs.html", help="Page URL to scan for .mid links")
    p.add_argument("--out", default="midis", help="Output directory (created if missing)")
    p.add_argument("--delay", type=float, default=0.3, help="Delay (s) between downloads")
    p.add_argument("--limit", type=int, default=0, help="Max number of files to download (0 = no limit)")
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {args.url} for .mid links...")
    links = find_mid_links(args.url)
    print(f"Found {len(links)} .mid links (will skip existing files).")

    count = 0
    for link in links:
        if args.limit and count >= args.limit:
            break
        ok, reason = download_file(link, out_dir, delay=args.delay)
        if ok:
            print(f"Downloaded: {os.path.basename(link)}")
            count += 1
        else:
            if reason == "exists":
                print(f"Skipping (exists): {os.path.basename(link)}")
            else:
                print(f"Failed: {os.path.basename(link)} -> {reason}")

    print(f"Done. {count} new files downloaded to {out_dir}.")


if __name__ == "__main__":
    main()
