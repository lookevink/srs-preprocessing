import javabridge
import bioformats
from pathlib import Path
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class OIRConverter:
    def __init__(self):
        """Initialize the Java VM for bioformats"""
        logger.info("Initializing Java VM...")
        javabridge.start_vm(class_path=bioformats.JARS)
        logger.info("Java VM initialized successfully")

    def __del__(self):
        """Clean up Java VM on object destruction"""
        try:
            javabridge.kill_vm()
            logger.info("Java VM shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down Java VM: {str(e)}")

    def extract_metadata(self, metadata, reader):
        # (Implementation remains the same)
        # ...
        pass  # Assuming code remains unchanged for brevity

    def convert(self, input_paths, output_dir):
        """
        Convert one or more OIR files to individual OME-TIFF files and save metadata

        Args:
            input_paths: List[str or Path] - Paths to input OIR files
            output_dir: str or Path - Directory where output OME-TIFFs will be saved
        """
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert all paths to strings
        input_paths = [str(p) for p in input_paths]

        try:
            for input_path in input_paths:
                # Generate output paths
                input_filename = Path(input_path).stem
                output_path = output_dir / f"{input_filename}.ome.tiff"
                metadata_path = output_path.with_suffix('.metadata.txt')

                logger.info(f"Converting {input_path} to {output_path}")
                logger.info(f"Metadata will be saved to {metadata_path}")

                # Create metadata store
                logger.debug("Creating metadata store...")
                metadata = javabridge.JClassWrapper(
                    'loci.formats.MetadataTools').createOMEXMLMetadata()

                # Initialize reader
                logger.debug("Creating image reader...")
                reader = javabridge.JClassWrapper('loci.formats.ImageReader')()
                reader.setMetadataStore(metadata)
                reader.setId(input_path)

                # Extract and save metadata
                logger.info("Extracting metadata...")
                metadata_store = {
                    'timestamp': datetime.now().isoformat(),
                    'input_file': input_path,
                    'output_file': str(output_path),
                    'metadata': self.extract_metadata(metadata, reader)
                }

                # Save metadata to file
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_store, f, indent=2)
                logger.info("Metadata saved successfully")

                logger.debug("Reading image dimensions...")
                size_x = reader.getSizeX()
                size_y = reader.getSizeY()
                size_c = reader.getSizeC()
                size_z = reader.getSizeZ()
                size_t = reader.getSizeT()

                logger.debug(f"Image dimensions: {size_x}x{size_y}, {
                             size_c} channels, {size_z} z-planes, {size_t} timepoints")

                # Set up writer
                logger.debug("Setting up writer...")
                writer = javabridge.JClassWrapper('loci.formats.ImageWriter')()
                writer.setMetadataRetrieve(metadata)
                writer.setId(str(output_path))

                # Read and write each plane
                plane_count = reader.getImageCount()
                for index in range(plane_count):
                    img = reader.openBytes(index)
                    writer.saveBytes(index, img)

                writer.close()
                reader.close()
                logger.info(f"Conversion of {
                            input_path} completed successfully")

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def cleanup(self):
        """Clean up Java VM on object destruction"""
        try:
            javabridge.kill_vm()
            logger.info("Java VM shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down Java VM: {str(e)}")
