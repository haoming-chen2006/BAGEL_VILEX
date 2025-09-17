#!/usr/bin/env python3
import requests
import zipfile
import io
import os

def main():
    url = "https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip"
    output_dir = os.path.join(os.path.dirname(__file__), "dat_examples")

    # Make sure target dir exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading from {url} ...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    print("Extracting ZIP contents ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(output_dir)
        print(f"Extracted {len(zf.namelist())} files to {output_dir}")

if __name__ == "__main__":
    main()
