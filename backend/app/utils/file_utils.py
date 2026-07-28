import hashlib
import re
from pathlib import Path

def sanitize_filename(filename: str) -> str:
    """Strips path traversal sequences and dangerous characters."""
    base_name = Path(filename).name
    # Replace non-alphanumeric chars (except dots, dashes, underscores) with '_'
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", base_name)
    return clean_name

def calculate_sha256(file_path: Path) -> str:
    """Computes the SHA-256 hash of a file by reading in chunks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()