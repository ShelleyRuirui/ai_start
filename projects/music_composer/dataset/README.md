Dataset preparation:

  - jazz midi files: downloaded from https://github.com/SaiKayala/Jazz-Ml-Dataset/blob/master/Jazz-Midi.tar.xz

  - blues midi files: wrote a download script to download midi files from relevant website.


MID downloader

This folder contains a small script to download all .mid files linked from the Blues MIDIs page on midkar.com.

Usage

Run from the repository root or this folder:

```bash
python dataset/download_midkar_blues.py
```

Options:

- `--url` : page to scan (default: https://midkar.com/Blues/Blues_MIDIs.html)
- `--out` : relative output folder inside `dataset/` (default: `midis`)
- `--delay`: seconds between downloads (default: 0.3)
- `--limit`: max number of files to download (0 = no limit)

Requirements

Install dependencies with:

```bash
pip install -r dataset/requirements.txt
```

Notes

- The script skips files that already exist in the output folder.
- Be polite: don't hammer the remote server; use `--delay` if you run many downloads.
