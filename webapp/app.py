"""
Visualization Machine Learning - Interactive Web Application
A learning journey from mathematical foundations to deep learning.
"""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Visualization Machine Learning")

# Project root (parent of webapp/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mount static assets
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Mount project images (all existing PNGs/GIFs)
app.mount("/images", StaticFiles(directory=PROJECT_ROOT), name="images")


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/source/{path:path}")
async def get_source(path: str):
    """Return source code of a Python script for display."""
    file_path = PROJECT_ROOT / path
    if file_path.suffix == ".py" and file_path.exists() and file_path.is_relative_to(PROJECT_ROOT):
        return {"code": file_path.read_text(encoding="utf-8", errors="replace")}
    return {"error": "File not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
