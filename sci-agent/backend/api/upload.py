import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

MAX_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    orig = Path(file.filename or "file")
    safe = f"{uuid.uuid4().hex[:8]}_{orig.name}"
    dest = WORKSPACE / safe

    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_SIZE:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_SIZE // (1024*1024)} MB")
            out.write(chunk)

    return {
        "filename": safe,
        "original_filename": orig.name,
        "path": str(dest),
        "size": size,
    }
