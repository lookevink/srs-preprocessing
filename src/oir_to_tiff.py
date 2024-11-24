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
        """Extract metadata from the image and return as dictionary"""
        metadata_store = {}

        def safe_get_value(value):
            """Safely convert Java objects to Python values"""
            if value is None:
                return None
            try:
                # Try converting to string first
                str_val = str(value)
                # Try converting to float if it looks like a number
                try:
                    if '.' in str_val or 'E' in str_val.upper():
                        return float(str_val)
                    elif str_val.isdigit():
                        return int(str_val)
                    return str_val
                except ValueError:
                    return str_val
            except Exception:
                return None

        try:
            # Get basic image information
            metadata_store['basic_info'] = {
                'size_x': int(reader.getSizeX()),
                'size_y': int(reader.getSizeY()),
                'size_c': int(reader.getSizeC()),
                'size_z': int(reader.getSizeZ()),
                'size_t': int(reader.getSizeT()),
                'dimension_order': str(reader.getDimensionOrder()),
                'is_rgb': bool(reader.isRGB()),
                'is_interleaved': bool(reader.isInterleaved()),
                'is_indexed': bool(reader.isIndexed()),
                'pixel_type': str(reader.getPixelType())
            }

            # Get original metadata as key-value pairs
            original_metadata = {}
            metadata_list = reader.getSeriesMetadata()
            if metadata_list is not None:
                # Convert Java HashMap to Python dict
                keys = metadata_list.keySet()
                iterator = keys.iterator()
                while iterator.hasNext():
                    key = str(iterator.next())
                    if not 'LUT' in key:   # No need for LUT data
                        value = metadata_list.get(key)
                        original_metadata[key] = safe_get_value(value)
            metadata_store['original_metadata'] = original_metadata

            # Get series count and metadata for each series
            series_count = reader.getSeriesCount()
            metadata_store['series'] = []

            for series in range(series_count):
                reader.setSeries(series)
                series_metadata = {
                    'series_number': series,
                    'image_name': safe_get_value(metadata.getImageName(series)),
                    'acquisition_date': safe_get_value(metadata.getImageAcquisitionDate(series)),
                    'description': safe_get_value(metadata.getImageDescription(series)),
                    'channels': []
                }

                # Get channel information
                for c in range(reader.getSizeC()):
                    channel = {
                        'name': safe_get_value(metadata.getChannelName(series, c)),
                        'emission_wavelength': safe_get_value(metadata.getChannelEmissionWavelength(series, c)),
                        'excitation_wavelength': safe_get_value(metadata.getChannelExcitationWavelength(series, c)),
                        'pinhole_size': safe_get_value(metadata.getChannelPinholeSize(series, c)),
                        'color': safe_get_value(metadata.getChannelColor(series, c)),
                        'samples_per_pixel': safe_get_value(metadata.getChannelSamplesPerPixel(series, c))
                    }

                    # Get detector information
                    detector_id = metadata.getDetectorID(series, c)
                    if detector_id is not None:
                        channel['detector'] = {
                            'id': safe_get_value(detector_id),
                            'model': safe_get_value(metadata.getDetectorModel(series, c)),
                            'type': safe_get_value(metadata.getDetectorType(series, c)),
                            'gain': safe_get_value(metadata.getDetectorGain(series, c)),
                            'offset': safe_get_value(metadata.getDetectorOffset(series, c)),
                            'voltage': safe_get_value(metadata.getDetectorVoltage(series, c))
                        }

                    # Get objective information
                    objective_id = metadata.getObjectiveID(series, c)
                    if objective_id is not None:
                        channel['objective'] = {
                            'id': safe_get_value(objective_id),
                            'model': safe_get_value(metadata.getObjectiveModel(series, c)),
                            'nominal_magnification': safe_get_value(metadata.getObjectiveNominalMagnification(series, c)),
                            'lens_na': safe_get_value(metadata.getObjectiveLensNA(series, c)),
                            'immersion': safe_get_value(metadata.getObjectiveImmersion(series, c)),
                            'working_distance': safe_get_value(metadata.getObjectiveWorkingDistance(series, c))
                        }

                    series_metadata['channels'].append(channel)

                # Get physical sizes
                series_metadata['physical_sizes'] = {
                    'physical_size_x': safe_get_value(metadata.getPixelsPhysicalSizeX(series)),
                    'physical_size_y': safe_get_value(metadata.getPixelsPhysicalSizeY(series)),
                    'physical_size_z': safe_get_value(metadata.getPixelsPhysicalSizeZ(series))
                }

                metadata_store['series'].append(series_metadata)

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            logger.exception("Full traceback:")

        return metadata_store

    def convert(self, input_paths, output_path):
        """
        Convert one or more OIR files to a single OME-TIFF file and save metadata

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

        # Create metadata output path
        metadata_path = str(Path(output_path).with_suffix('.metadata.txt'))

        logger.info(f"Converting {len(input_paths)} file(s) to {output_path}")
        logger.info(f"Metadata will be saved to {metadata_path}")

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

            # Extract LUT data separately
            lut_metadata = {}
            metadata_list = reader.getSeriesMetadata()
            if metadata_list is not None:
                keys = metadata_list.keySet()
                iterator = keys.iterator()
                while iterator.hasNext():
                    key = str(iterator.next())
                    if 'LUT' in key:
                        value = metadata_list.get(key)
                        lut_metadata[key] = value
                        logger.debug(f"Found LUT data: {key}")

            # Store LUT data in OME-XML metadata if present
            if lut_metadata:
                logger.info(f"Found {len(lut_metadata)} LUT entries")
                for series in range(reader.getSeriesCount()):
                    reader.setSeries(series)
                    for c in range(reader.getSizeC()):
                        # Store LUT data in channel metadata
                        lut_key = f"LUT_S{series}_C{c}"
                        if lut_key in lut_metadata:
                            logger.debug(f"Setting LUT data for series {
                                         series}, channel {c}")
                            # Store as original metadata since OME-XML doesn't have a standard LUT field
                            metadata.setChannelID(
                                f"Channel:S{series}:C{c}", series, c)
                            # Note: This is a simplified approach - the actual LUT data remains in the metadata file
                            # rather than being embedded in the OME-TIFF

            # Extract and save metadata
            logger.info("Extracting metadata...")
            metadata_store = {
                'timestamp': datetime.now().isoformat(),
                'input_files': input_paths,
                'output_file': output_path,
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
