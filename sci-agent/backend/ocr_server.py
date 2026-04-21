"""Standalone Nemotron OCR V2 server (English).

Run: `python3 -m backend.ocr_server`
Port: $SCI_AGENT_OCR_PORT (default 8788)

Nemotron OCR V2 only accepts raster images (jpeg/png/webp/gif). For PDF input
we rasterize each page via PyMuPDF first and run OCR per page.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

app = FastAPI(title="Nemotron OCR (en)")

ocr = NemotronOCRV2(lang="en")

# Nemotron OCR-supported raster formats (see torchvision decode_image).
_RASTER_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_PDF_EXT = ".pdf"


def _ocr_image(path: str, merge_level: str) -> Any:
    with torch.inference_mode():
        return ocr(path, merge_level=merge_level)


def _ocr_pdf(pdf_path: str, merge_level: str) -> list:
    """Rasterize each page to PNG, OCR, stamp page number, flatten."""
    if not _HAS_FITZ:
        raise RuntimeError(
            "PyMuPDF not installed. `pip install pymupdf` in the nemotron-ocr env."
        )

    all_preds: list = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx, page in enumerate(doc, start=1):
            # 2x zoom for OCR fidelity.
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                pix.save(tmp_path)
            try:
                preds = _ocr_image(tmp_path, merge_level)
            finally:
                os.unlink(tmp_path)

            if isinstance(preds, list):
                for p in preds:
                    if isinstance(p, dict):
                        p["page"] = page_idx
                all_preds.extend(preds)
            elif preds:
                all_preds.append({"page": page_idx, "text": str(preds)})
    finally:
        doc.close()
    return all_preds


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pdf_support": _HAS_FITZ,
    }


@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    merge_level: str = "paragraph",
):
    orig_name = file.filename or "image.png"
    ext = Path(orig_name).suffix.lower() or ".png"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        if ext == _PDF_EXT:
            if not _HAS_FITZ:
                return JSONResponse(
                    status_code=501,
                    content={
                        "error": "PDF support requires PyMuPDF. "
                        "Run `pip install pymupdf` in the nemotron-ocr env."
                    },
                )
            result = _ocr_pdf(tmp.name, merge_level=merge_level)
        elif ext in _RASTER_EXTS:
            result = _ocr_image(tmp.name, merge_level=merge_level)
        else:
            return JSONResponse(
                status_code=415,
                content={
                    "error": f"Unsupported file type '{ext}'. "
                    "Supported: PDF, JPEG, PNG, WebP, GIF."
                },
            )

    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SCI_AGENT_OCR_PORT", "8788"))
    reload = os.environ.get("SCI_AGENT_OCR_RELOAD", "0") == "1"
    uvicorn.run(
        "backend.ocr_server:app",
        host="127.0.0.1",
        port=port,
        reload=reload,
    )
