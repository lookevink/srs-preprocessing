from oir_to_tiff import OIRConverter
from pathlib import Path
import tifffile
import numpy as np


def main():
    # Input and output paths
    input_paths = [
        Path("data/input/793_102.18to101.68_6steps01_20_30_2k_1.5offset.oir"),
        Path("data/input/804_101.3to100.4_11steps009_20_30_2k_1.5offset.oir"),
        Path("data/input/886_91.98to91.18_9steps01_20_30_2k_1.5offset.oir"),
        Path("data/input/897_90.84to90.14_9steps00875_20_30_2k_1.5offset.oir")
    ]
    output_dir = Path("data/output")
    temp_output_dir = output_dir / "temp"
    merged_output_path = output_dir / "merged.ome.tiff"

    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize converter
        converter = OIRConverter()

        # Convert files to OME-TIFFs in temp directory
        print(f"Converting {len(input_paths)
                            } files to OME-TIFF in temp directory...")
        converter.convert(input_paths, temp_output_dir)
        print(f"Successfully converted files to OME-TIFF in {temp_output_dir}")

        # Now, merge the OME-TIFF files into a single OME-TIFF file
        print(f"Merging OME-TIFF files into {merged_output_path}...")

        # Collect all OME-TIFF files in temp_output_dir
        ome_tiff_paths = sorted(temp_output_dir.glob("*.ome.tiff"))

        # Read images and collect data
        images = []
        for ome_tiff_path in ome_tiff_paths:
            print(f"Reading {ome_tiff_path}")
            with tifffile.TiffFile(str(ome_tiff_path)) as tif:
                # Read images
                img = tif.asarray()
                images.append(img)

        # Concatenate images along a new axis (e.g., time axis)
        merged_images = np.concatenate(images, axis=0)

        # Save merged OME-TIFF
        print(f"Saving merged OME-TIFF to {merged_output_path}")
        with tifffile.TiffWriter(str(merged_output_path), bigtiff=True, ome=True) as tif:
            tif.write(merged_images, photometric='minisblack',
                      metadata={'axes': 'TZCYX'})

        print(f"Successfully merged OME-TIFF files to {merged_output_path}")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    main()
