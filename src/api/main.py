from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import asyncio
import tifffile
import numpy as np
import javabridge
import bioformats
import logging
from concurrent.futures import ThreadPoolExecutor

from src.conversion.oir_to_tiff import OIRConverter
from src.stabilize.stabilizer_optical_flow import ImageStabilizer
from src.stabilize.stabilizer_ransac_offset import RANSACStabilizer

# Setup logging at the top of the file
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SRS Image Processing API")

# Create temporary directories for processing
TEMP_INPUT_DIR = Path("data/temp/input")
TEMP_OUTPUT_DIR = Path("data/temp/output")
TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add static files mounting AFTER FastAPI initialization but BEFORE routes
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# Create a thread pool executor for blocking operations
thread_pool = ThreadPoolExecutor(max_workers=4)


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

    try:
        # Save uploaded files first
        input_paths = []
        for file in files:
            if not file.filename.lower().endswith('.oir'):
                raise HTTPException(
                    status_code=400, detail="Invalid file format")

            file_path = session_input / file.filename
            # Read in chunks to prevent memory issues
            with file_path.open('wb') as buffer:
                while content := await file.read(8192):  # 8KB chunks
                    buffer.write(content)
            input_paths.append(file_path)

        # Process files in thread pool
        converter = OIRConverter()

        # Run conversion in thread pool
        await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: converter.convert(input_paths, session_output)
        )

        if stabilize:
            # Process stabilization if requested
            tiff_files = list(session_output.glob("*.ome.tiff"))
            merged_path = session_output / "merged.ome.tiff"

            # Read and merge images in thread pool
            def process_images():
                images = []
                axes_list = []
                for tiff_path in sorted(tiff_files):
                    with tifffile.TiffFile(str(tiff_path)) as tif:
                        series = tif.series[0]
                        img = series.asarray()
                        axes = series.axes
                        images.append(img)
                        axes_list.append(axes)
                return images, axes_list

            images, axes_list = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                process_images
            )

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

            return FileResponse(
                path=final_output,
                filename="stabilized.ome.tiff",
                media_type="image/tiff",
                background=cleanup_file(final_output)
            )
        else:
            # Create zip of results
            output_zip = TEMP_OUTPUT_DIR / f"{session_id}.zip"
            await asyncio.to_thread(
                lambda: shutil.make_archive(
                    str(output_zip.with_suffix('')),
                    'zip',
                    session_output
                )
            )

            return FileResponse(
                path=output_zip,
                filename="converted_files.zip",
                media_type="application/zip",
                background=cleanup_file(output_zip)
            )

    except Exception as e:
        logger.error(f"Error in convert_oir_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        asyncio.create_task(cleanup_directories(session_input, session_output))


async def cleanup_directories(input_dir: Path, output_dir: Path):
    """Clean up temporary directories asynchronously"""
    await asyncio.sleep(1)  # Give time for file operations to complete
    try:
        if input_dir.exists():
            await asyncio.to_thread(lambda: shutil.rmtree(input_dir))
        if output_dir.exists():
            await asyncio.to_thread(lambda: shutil.rmtree(output_dir))
    except Exception as e:
        logger.error(f"Error cleaning up directories: {e}")


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

    # Don't initialize VM here - let OIRConverter handle it


@app.on_event("shutdown")
async def shutdown_event():
    # Only kill VM if it's actually running
    if OIRConverter._vm_initialized:
        try:
            javabridge.kill_vm()
            OIRConverter._vm_initialized = False
        except Exception as e:
            logger.error(f"Error shutting down Java VM: {str(e)}")


# Add this helper function at the module level
def cleanup_file(file_path: Path):
    """Clean up a file after it has been sent"""
    async def _cleanup():
        try:
            await asyncio.sleep(1)  # Give time for the file to be sent
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")
    return _cleanup
