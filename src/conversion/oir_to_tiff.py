import javabridge
import bioformats
from pathlib import Path
import logging
import json
from datetime import datetime
import threading

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class OIRConverter:
    _instance = None
    _vm_initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        logger.debug("Creating new OIRConverter instance")
        if cls._instance is None:
            cls._instance = super(OIRConverter, cls).__new__(cls)
            logger.debug("Created new OIRConverter instance")
        else:
            logger.debug("Returning existing OIRConverter instance")
        return cls._instance

    def __init__(self):
        logger.debug("Initializing OIRConverter")
        self.ensure_vm_running()

    def ensure_vm_running(self):
        """Ensure VM is running before operations"""
        logger.debug(
            f"Checking VM state. Current state: {OIRConverter._vm_initialized}")

        if not OIRConverter._vm_initialized:
            with self._lock:  # Only lock during initialization
                logger.debug("Acquired lock for VM initialization")
                if not OIRConverter._vm_initialized:  # Double-check pattern
                    try:
                        logger.debug("Starting Java VM...")
                        javabridge.start_vm(class_path=bioformats.JARS,
                                            run_headless=True,
                                            max_heap_size='4G')

                        # Attach to thread and verify VM is responsive
                        self._attach_thread()
                        env = javabridge.get_env()
                        if env is None:
                            raise RuntimeError(
                                "Java VM environment is None after initialization")

                        # Test basic Java operation
                        test_class = javabridge.JClassWrapper(
                            'java.lang.String')
                        if test_class is None:
                            raise RuntimeError(
                                "Cannot access Java classes after VM initialization")

                        OIRConverter._vm_initialized = True
                        logger.debug("Java VM initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize Java VM: {str(e)}")
                        OIRConverter._vm_initialized = False
                        raise RuntimeError(
                            f"Failed to initialize Java VM: {str(e)}")
                    finally:
                        logger.debug(
                            "Releasing lock after VM initialization attempt")

    def _attach_thread(self):
        """Attach current thread to JVM"""
        try:
            logger.debug("Attempting to attach thread to JVM")
            if javabridge.get_env() is None:
                javabridge.attach()
                logger.debug("Successfully attached thread to JVM")
            else:
                logger.debug("Thread already attached to JVM")
        except Exception as e:
            logger.error(f"Error attaching thread to JVM: {str(e)}")
            raise

    def extract_metadata(self, metadata, reader):
        # (Implementation remains the same)
        # ...
        pass  # Assuming code remains unchanged for brevity

    def convert(self, input_paths, output_dir):
        """Convert OIR files to OME-TIFF format"""
        logger.debug(f"Starting conversion of {len(input_paths)} files")
        self.ensure_vm_running()

        # Attach thread at the start of conversion
        self._attach_thread()

        for input_path in input_paths:
            try:
                logger.debug(f"Processing file: {input_path}")

                # Generate output paths
                input_filename = Path(input_path).stem
                output_path = Path(output_dir) / f"{input_filename}.ome.tiff"
                metadata_path = output_path.with_suffix('.metadata.txt')

                logger.debug(f"Converting {input_path} to {output_path}")

                # Verify VM state before creating Java objects
                logger.debug("Verifying VM state before conversion")
                env = javabridge.get_env()
                if env is None:
                    logger.debug(
                        "VM environment is None, attempting to reattach")
                    self._attach_thread()
                    env = javabridge.get_env()
                    if env is None:
                        raise RuntimeError(
                            "Java VM environment is None before conversion")

                # Create metadata store and reader outside of lock
                logger.debug("Creating metadata store and reader")
                metadata = javabridge.JClassWrapper(
                    'loci.formats.MetadataTools').createOMEXMLMetadata()
                reader = javabridge.JClassWrapper('loci.formats.ImageReader')()

                with self._lock:  # Lock only critical Java operations
                    logger.debug("Acquired lock for Java operations")
                    reader.setMetadataStore(metadata)
                    reader.setId(str(input_path))

                    # Extract dimensions
                    size_x = reader.getSizeX()
                    size_y = reader.getSizeY()
                    size_c = reader.getSizeC()
                    size_z = reader.getSizeZ()
                    size_t = reader.getSizeT()

                    logger.debug(
                        f"Image dimensions: {size_x}x{size_y}, C={size_c}, Z={size_z}, T={size_t}")

                    # Set up writer
                    writer = javabridge.JClassWrapper(
                        'loci.formats.ImageWriter')()
                    writer.setMetadataRetrieve(metadata)
                    writer.setId(str(output_path))

                    # Read and write each plane
                    plane_count = reader.getImageCount()
                    logger.debug(f"Processing {plane_count} planes")

                    for index in range(plane_count):
                        logger.debug(
                            f"Processing plane {index+1}/{plane_count}")
                        img = reader.openBytes(index)
                        writer.saveBytes(index, img)

                    writer.close()
                    reader.close()
                    logger.debug("Closed writer and reader")

                logger.debug(f"Successfully converted {input_path}")

            except Exception as e:
                logger.error(f"Error converting file {input_path}: {str(e)}")
                logger.exception("Detailed error trace:")
                raise
            finally:
                logger.debug("Finished processing file")

    def cleanup(self):
        """Clean up Java VM"""
        logger.debug("Attempting to clean up Java VM")
        if OIRConverter._vm_initialized:
            with self._lock:
                logger.debug("Acquired lock for VM cleanup")
                if OIRConverter._vm_initialized:
                    try:
                        # Detach current thread before killing VM
                        try:
                            javabridge.detach()
                            logger.debug("Detached thread from JVM")
                        except:
                            logger.debug("Thread was not attached to JVM")

                        javabridge.kill_vm()
                        OIRConverter._vm_initialized = False
                        logger.debug("Java VM shut down successfully")
                    except Exception as e:
                        logger.error(f"Error shutting down Java VM: {str(e)}")
                    finally:
                        logger.debug("Released lock after VM cleanup attempt")
