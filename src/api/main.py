from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import asyncio
import tifffile
import numpy as np

from src.conversion.oir_to_tiff import OIRConverter
from src.stabilize.stabilizer_optical_flow import ImageStabilizer
from src.stabilize.stabilizer_ransac_offset import RANSACStabilizer

app = FastAPI(title="SRS Image Processing API")

# Create temporary directories for processing
TEMP_INPUT_DIR = Path("data/temp/input")
TEMP_OUTPUT_DIR = Path("data/temp/output")
TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add static files mounting AFTER FastAPI initialization but BEFORE routes
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.post("/convert-oir")
async def convert_oir_files(
    files: list[UploadFile] = File(...),
    stabilize: bool = False,
    method: str = "optical_flow"
):
    """Convert OIR files to OME-TIFF format and optionally stabilize"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create session directory
    session_id = str(uuid.uuid4())
    session_input = TEMP_INPUT_DIR / session_id
    session_output = TEMP_OUTPUT_DIR / session_id
    session_input.mkdir(parents=True)
    session_output.mkdir(parents=True)

    converter = None
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

        if stabilize:
            # Process all files before cleaning up Java VM
            tiff_files = list(session_output.glob("*.ome.tiff"))

            # Merge converted files into single TIFF
            merged_path = session_output / "merged.ome.tiff"

            # Read images and collect data
            images = []
            axes_list = []
            for tiff_path in sorted(tiff_files):
                with tifffile.TiffFile(str(tiff_path)) as tif:
                    series = tif.series[0]
                    img = series.asarray()
                    axes = series.axes
                    images.append(img)
                    axes_list.append(axes)

            # Now we can cleanup the Java VM as we have all the data we need
            if converter:
                converter.cleanup()
                converter = None

            # Validate axes consistency
            base_axes = axes_list[0]
            for axes in axes_list[1:]:
                if axes != base_axes:
                    raise HTTPException(
                        status_code=400, detail="Not all images have the same axes labels")

            # Find T axis
            t_index = base_axes.find('T')
            if t_index == -1:
                raise HTTPException(
                    status_code=400, detail="Axes labels do not contain 'T' axis")

            # Validate shapes
            base_shape = images[0].shape
            for img in images[1:]:
                shape_without_t = [s for idx, s in enumerate(
                    base_shape) if idx != t_index]
                img_shape_without_t = [
                    s for idx, s in enumerate(img.shape) if idx != t_index]
                if shape_without_t != img_shape_without_t:
                    raise HTTPException(
                        status_code=400, detail="Not all images have the same shape except for T dimension")

            # Merge along T axis
            merged_image = np.concatenate(images, axis=t_index)

            # Save merged file
            with tifffile.TiffWriter(str(merged_path), bigtiff=True, ome=True) as tif:
                tif.write(merged_image, photometric='minisblack',
                          metadata={'axes': base_axes})

            # Stabilize merged file
            stabilized_path = session_output / "stabilized.ome.tiff"
            print(
                f"Attempting to stabilize {merged_path} to {stabilized_path}")

            if method == "optical_flow":
                stabilizer = ImageStabilizer(method=method)
                print(f"Created stabilizer with method: {method}")

                # Verify merged file exists
                print(f"Merged file exists: {merged_path.exists()}")
                print(
                    f"Merged file size: {merged_path.stat().st_size if merged_path.exists() else 'N/A'}")

                # Use stabilize_file method
                stabilizer.stabilize_file(merged_path, stabilized_path)

                # Verify stabilized file was created
                print(f"Stabilized file exists: {stabilized_path.exists()}")
                print(
                    f"Stabilized file size: {stabilized_path.stat().st_size if stabilized_path.exists() else 'N/A'}")

            elif method == "ransac":
                stabilizer = RANSACStabilizer(merged_path, stabilized_path)
                stabilizer.stabilize()
            else:
                raise HTTPException(
                    status_code=400, detail="Invalid stabilization method")

            # Verify file exists before returning
            if not stabilized_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Stabilization failed to create output file at {stabilized_path}"
                )

            # Copy the stabilized file to a new location outside the temp directory
            final_output = TEMP_OUTPUT_DIR / \
                f"stabilized_{session_id}.ome.tiff"
            shutil.copy2(stabilized_path, final_output)

            # Clean up temporary directories
            if session_input.exists():
                shutil.rmtree(session_input)
            if session_output.exists():
                shutil.rmtree(session_output)

            return FileResponse(
                path=final_output,
                filename="stabilized.ome.tiff",
                media_type="image/tiff",
                background=asyncio.create_task(
                    cleanup_file(final_output)
                )
            )

        else:
            # Cleanup Java VM before creating zip
            if converter:
                converter.cleanup()
                converter = None

            # Create zip of results if not stabilizing
            output_zip = TEMP_OUTPUT_DIR / f"{session_id}.zip"
            shutil.make_archive(str(output_zip.with_suffix('')),
                                'zip', session_output)

            return FileResponse(
                path=output_zip,
                filename="converted_files.zip",
                media_type="application/zip"
            )

    except Exception as e:
        if converter:
            converter.cleanup()
        # Clean up temporary directories
        if session_input.exists():
            shutil.rmtree(session_input)
        if session_output.exists():
            shutil.rmtree(session_output)
        raise e

    finally:
        # Remove cleanup from finally block
        pass


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
            stabilizer = ImageStabilizer(method=method)
            stabilizer.stabilize_file(input_path, output_path)
        elif method == "ransac":
            stabilizer = RANSACStabilizer(input_path, output_path)
            stabilizer.stabilize()
        else:
            raise HTTPException(
                status_code=400, detail="Invalid stabilization method")

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


# Add this helper function at the module level
async def cleanup_file(file_path: Path):
    """Clean up a file after it has been sent"""
    try:
        await asyncio.sleep(1)  # Give time for the file to be sent
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")
