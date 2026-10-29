import os
import io
import struct
import zlib
import urllib.request
from typing import Dict, Optional, List
from PIL import Image

class RemoteZipIndex:
    """Manages selective downloading of files from a remote zip using HTTP Range requests."""
    
    def __init__(self, url: str, timeout: int = 15):
        self.url = url
        self.timeout = timeout
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.length = int(resp.headers['Content-Length'])
        self.entries: Dict[str, Dict] = {}
        self._load_central_directory()

    def _read_range(self, start: int, size: int) -> bytes:
        end = start + size - 1
        req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def _load_central_directory(self):
        tail_size = min(65536, self.length)
        tail_data = self._read_range(self.length - tail_size, tail_size)
        eocd_idx = tail_data.rfind(b'\x50\x4b\x05\x06')
        if eocd_idx == -1:
            raise ValueError(f"End of Central Directory not found in {self.url}")
            
        eocd = tail_data[eocd_idx:eocd_idx+22]
        _, _, _, total_entries, cd_size, cd_offset = struct.unpack('<HHHHII', eocd[4:20])
        cd_data = self._read_range(cd_offset, cd_size)
        
        pos = 0
        while pos < len(cd_data):
            if cd_data[pos:pos+4] != b'\x50\x4b\x01\x02':
                break
            compression = struct.unpack('<H', cd_data[pos+10:pos+12])[0]
            comp_size = struct.unpack('<I', cd_data[pos+20:pos+24])[0]
            uncomp_size = struct.unpack('<I', cd_data[pos+24:pos+28])[0]
            fn_len = struct.unpack('<H', cd_data[pos+28:pos+30])[0]
            extra_len = struct.unpack('<H', cd_data[pos+30:pos+32])[0]
            comment_len = struct.unpack('<H', cd_data[pos+32:pos+34])[0]
            local_offset = struct.unpack('<I', cd_data[pos+42:pos+46])[0]
            fn = cd_data[pos+46:pos+46+fn_len].decode('utf-8', errors='ignore')
            
            # Map filename without leading directory
            base_fn = os.path.basename(fn)
            self.entries[base_fn] = {
                'full_path': fn,
                'compression': compression,
                'comp_size': comp_size,
                'uncomp_size': uncomp_size,
                'local_offset': local_offset
            }
            pos += 46 + fn_len + extra_len + comment_len

    def get_image_bytes(self, filename: str) -> bytes:
        base_fn = os.path.basename(filename)
        if base_fn not in self.entries:
            raise KeyError(f"Image {filename} not found in remote archive {self.url}")
            
        info = self.entries[base_fn]
        header_data = self._read_range(info['local_offset'], 30)
        local_fn_len = struct.unpack('<H', header_data[26:28])[0]
        local_extra_len = struct.unpack('<H', header_data[28:30])[0]
        data_offset = info['local_offset'] + 30 + local_fn_len + local_extra_len
        raw_data = self._read_range(data_offset, info['comp_size'])
        
        if info['compression'] == 0:
            return raw_data
        elif info['compression'] == 8:
            return zlib.decompress(raw_data, -15)
        else:
            raise ValueError(f"Unsupported compression {info['compression']} for {filename}")

class DatasetCacheManager:
    """Caches images locally and fetches on demand from remote archive or local disk."""
    
    def __init__(self, cache_dir: str, remote_url: Optional[str] = None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.remote_url = remote_url
        self._remote_zip: Optional[RemoteZipIndex] = None

    def _get_remote(self) -> RemoteZipIndex:
        if self._remote_zip is None:
            if not self.remote_url:
                raise ValueError("No remote URL specified for DatasetCacheManager")
            self._remote_zip = RemoteZipIndex(self.remote_url)
        return self._remote_zip

    def get_image_path(self, filename: str) -> str:
        """Returns local path to cached image, downloading it if not present."""
        base_fn = os.path.basename(filename)
        local_path = os.path.join(self.cache_dir, base_fn)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
            
        # Download selectively
        remote = self._get_remote()
        data = remote.get_image_bytes(base_fn)
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path

    def get_image(self, filename: str) -> Image.Image:
        """Returns PIL Image for the requested filename."""
        path = self.get_image_path(filename)
        return Image.open(path).convert("RGB")
