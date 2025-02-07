from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import asyncio

from src.conversion.oir_to_tiff import OIRConverter
from src.stabilize.stabilizer_optical_flow import ImageStabilizer
from src.stabilize.stabilizer_ransac_offset import RANSACStabilizer

app = FastAPI(title="SRS Image Processing API")

# Create temporary directories for processing
TEMP_INPUT_DIR = Path("data/temp/input")
TEMP_OUTPUT_DIR = Path("data/temp/output")
TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/convert-oir")
async def convert_oir_files(files: list[UploadFile]):
    """Convert OIR files to OME-TIFF format"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create session directory
    session_id = str(uuid.uuid4())
    session_input = TEMP_INPUT_DIR / session_id
    session_output = TEMP_OUTPUT_DIR / session_id
    session_input.mkdir(parents=True)
    session_output.mkdir(parents=True)

    try:
        # Save uploaded files
        input_paths = []
        for file in files:
            if not file.filename.lower().endswith('.oir'):
                raise HTTPException(
                    status_code=400, detail="Invalid file format. Only .oir files are accepted")

            file_path = session_input / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_paths.append(file_path)

        # Convert files
        converter = OIRConverter()
        converter.convert(input_paths, session_output)

        # Create zip of results
        output_zip = TEMP_OUTPUT_DIR / f"{session_id}.zip"
        shutil.make_archive(str(output_zip.with_suffix('')),
                            'zip', session_output)

        return FileResponse(
            path=output_zip,
            filename="converted_files.zip",
            media_type="application/zip"
        )

    finally:
        # Cleanup
        if session_input.exists():
            shutil.rmtree(session_input)
        if session_output.exists():
            shutil.rmtree(session_output)


@app.post("/stabilize")
async def stabilize_image(
    file: UploadFile,
    method: str = "optical_flow"  # or "ransac"
):
    """Stabilize an OME-TIFF image"""
    if not file.filename.lower().endswith(('.tiff', '.tif')):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Only TIFF files are accepted")

    # Create session directory
    session_id = str(uuid.uuid4())
    session_input = TEMP_INPUT_DIR / session_id
    session_output = TEMP_OUTPUT_DIR / session_id
    session_input.mkdir(parents=True)
    session_output.mkdir(parents=True)

    try:
        # Save uploaded file
        input_path = session_input / file.filename
        output_path = session_output / f"stabilized_{file.filename}"

        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Stabilize image
        if method == "optical_flow":
            stabilizer = ImageStabilizer(input_path, output_path)
        elif method == "ransac":
            stabilizer = RANSACStabilizer(input_path, output_path)
        else:
            raise HTTPException(
                status_code=400, detail="Invalid stabilization method")

        stabilizer.stabilize()

        return FileResponse(
            path=output_path,
            filename=f"stabilized_{file.filename}",
            media_type="image/tiff"
        )

    finally:
        # Cleanup
        if session_input.exists():
            shutil.rmtree(session_input)
        if session_output.exists():
            shutil.rmtree(session_output)


@app.on_event("startup")
async def startup_event():
    # Clean any existing temporary files
    if TEMP_INPUT_DIR.exists():
        shutil.rmtree(TEMP_INPUT_DIR)
    if TEMP_OUTPUT_DIR.exists():
        shutil.rmtree(TEMP_OUTPUT_DIR)

    TEMP_INPUT_DIR.mkdir(parents=True)
    TEMP_OUTPUT_DIR.mkdir(parents=True)
