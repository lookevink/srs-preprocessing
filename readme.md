# SRS Data Processing Pipeline

Simple data preprocessing pipeline for SRS data with REST API support.

## Prerequisites

1. Python 3.12 or higher
2. Java Runtime Environment (JRE) 8 or higher
   - Required for bioformats OIR file processing
   - Installation:
     - Ubuntu/Debian: `sudo apt-get install default-jre`
     - macOS: `brew install java`
     - Windows: Download and install from [java.com](https://www.java.com)

## Setup

1. Create a Python virtual environment:
```bash
python3.12 -m venv venv
```

2. Activate the virtual environment:
```bash
# On Linux/macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

1. Load the OIR files into `data/input/`
2. Run the pipeline:
```bash
python -m src.main
```
The output will be saved in `data/output/`. You can compare the unstable and stable outputs to see the difference.

### REST API

1. Start the API server:
```bash
python -m src.api.run
```

2. The API will be available at:
- API Endpoints: `http://localhost:8000`
- Interactive Documentation: `http://localhost:8000/docs`
- API Reference: `http://localhost:8000/redoc`

#### Available Endpoints

1. **Convert OIR Files** (`POST /convert-oir`)
   - Converts OIR files to OME-TIFF format
   - Accepts multiple files
   - Returns a ZIP file containing converted files
   ```bash
   curl -X POST "http://localhost:8000/convert-oir" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@/path/to/file1.oir" \
     -F "files=@/path/to/file2.oir" \
     --output converted_files.zip
   ```

2. **Stabilize Image** (`POST /stabilize`)
   - Stabilizes an OME-TIFF image
   - Supports two methods: optical_flow (default) and ransac
   - Returns the stabilized TIFF file
   ```bash
   # Using optical flow
   curl -X POST "http://localhost:8000/stabilize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.tiff" \
     --output stabilized_image.tiff

   # Using RANSAC
   curl -X POST "http://localhost:8000/stabilize?method=ransac" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.tiff" \
     --output stabilized_image.tiff
   ```

## Project Structure

```
.
├── data/
│   ├── input/      # Place input OIR files here
│   ├── output/     # Processed files will be saved here
│   └── temp/       # Temporary files for API processing
├── src/
│   ├── api/        # REST API implementation
│   ├── conversion/ # OIR to TIFF conversion
│   └── stabilize/  # Image stabilization algorithms
├── tests/          # Unit tests
├── requirements.txt
└── README.md
```

## Notes

- The API uses temporary directories for processing. These are automatically cleaned up after each request.
- Large files may take several minutes to process, especially during OIR conversion.
- For optimal performance, ensure sufficient RAM is available for processing large image stacks.

## Support

Don't hesitate to reach out if adjustments are needed. We can expand the scope of this to speed & scale up SRS-tailored spectral matching algorithm in general to increase the speed of the detection pipeline.

