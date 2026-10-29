import urllib.request
import zipfile
import io
import os

url = "https://vizwiz.cs.colorado.edu/dataset_downloads/VizWiz-Priv-v1.0/Annotations.zip"
print(f"Downloading {url}...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = resp.read()

print(f"Downloaded {len(data)} bytes. Extracting...")
os.makedirs("data/raw/Annotations", exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    z.printdir()
    z.extractall("data/raw/Annotations")

print("Done! Files extracted to data/raw/Annotations")
