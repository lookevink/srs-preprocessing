import javabridge
import bioformats
from pathlib import Path
import logging

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

    def convert(self, input_paths, output_path):
        """
        Convert one or more OIR files to a single OME-TIFF file

        Args:
            input_paths: str or Path - Path(s) to input OIR file(s)
            output_path: str or Path - Path where output OME-TIFF will be saved
        """
        # Convert single path to list
        if isinstance(input_paths, (str, Path)):
            input_paths = [input_paths]

        # Convert all paths to strings
        input_paths = [str(p) for p in input_paths]
        output_path = str(output_path)

        logger.info(f"Converting {len(input_paths)} file(s) to {output_path}")

        try:
            # Create metadata store
            logger.debug("Creating metadata store...")
            metadata = javabridge.JClassWrapper(
                'loci.formats.MetadataTools').createOMEXMLMetadata()

            # Initialize reader
            logger.debug("Creating image reader...")
            reader = javabridge.JClassWrapper('loci.formats.ImageReader')()
            reader.setMetadataStore(metadata)
            reader.setId(input_paths[0])

            # Get image dimensions
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
            writer.setId(output_path)

            # Process each input file
            for i, input_path in enumerate(input_paths):
                logger.info(f"Processing file {
                            i+1}/{len(input_paths)}: {input_path}")

                if i > 0:  # Only need to set ID for subsequent files
                    reader.setId(input_path)

                # Read all planes from the file
                for t in range(size_t):
                    for z in range(size_z):
                        for c in range(size_c):
                            index = z + (c * size_z) + (t * size_z * size_c)
                            logger.debug(f"Reading plane t={t}, z={z}, c={c}")
                            img = reader.openBytes(index)

                            # Write the plane
                            logger.debug(f"Writing plane index {index}")
                            writer.saveBytes(
                                index + (i * size_z * size_c * size_t),
                                img
                            )

            writer.close()
            reader.close()
            logger.info("Conversion completed successfully")

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            logger.exception("Full traceback:")
            raise
