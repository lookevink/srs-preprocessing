from pathlib import Path
import tifffile
import numpy as np

from src.conversion.oir_to_tiff import OIRConverter
from src.stabilize.stabilizer_optical_flow import ImageStabilizer


def main():
    # Input and output paths
    input_dir = Path("data/input")
    # Get all .oir files in input directory
    input_paths = sorted(input_dir.glob("*.oir"))
    if not input_paths:
        raise ValueError("No .oir files found in input directory")

    output_dir = Path("data/output")
    temp_output_dir = output_dir / "temp"
    merged_output_path = output_dir / "merged.ome.tiff"
    stabilized_output_path = output_dir / "stabilized.ome.tiff"

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
        axes_list = []
        for ome_tiff_path in ome_tiff_paths:
            print(f"Reading {ome_tiff_path}")
            with tifffile.TiffFile(str(ome_tiff_path)) as tif:
                series = tif.series[0]
                img = series.asarray()
                axes = series.axes  # Axes labels, e.g., 'TCZYX' or 'TZCYX'
                images.append(img)
                axes_list.append(axes)
                print(f"Image shape: {img.shape}, axes: {axes}")

        # Ensure all images have the same axes labels
        base_axes = axes_list[0]
        for axes in axes_list[1:]:
            if axes != base_axes:
                raise ValueError("Not all images have the same axes labels")

        # Find the index of 'T' in the axes labels
        t_index = base_axes.find('T')
        if t_index == -1:
            raise ValueError("Axes labels do not contain 'T' axis")

        # Ensure all images have the same shape except for the 'T' dimension
        base_shape = images[0].shape
        for img in images[1:]:
            shape_without_t = [s for idx, s in enumerate(
                base_shape) if idx != t_index]
            img_shape_without_t = [s for idx,
                                   s in enumerate(img.shape) if idx != t_index]
            if shape_without_t != img_shape_without_t:
                raise ValueError(
                    "Not all images have the same shape except for T dimension")

        # Concatenate images along the 'T' axis
        merged_images = np.concatenate(images, axis=t_index)
        print(f"Merged image shape: {merged_images.shape}, axes: {base_axes}")

        # Save merged OME-TIFF
        print(f"Saving merged OME-TIFF to {merged_output_path}")
        with tifffile.TiffWriter(str(merged_output_path), bigtiff=True, ome=True) as tif:
            tif.write(merged_images, photometric='minisblack',
                      metadata={'axes': base_axes})

        print(f"Successfully merged OME-TIFF files to {merged_output_path}")

        # Stabilize the merged image
        print("Stabilizing merged image...")
        stabilizer = ImageStabilizer(
            merged_output_path, stabilized_output_path)
        stabilizer.stabilize()
        print(f"Successfully stabilized image and saved to {
              stabilized_output_path}")

    except Exception as e:
        print(f"Error during conversion or stabilization: {str(e)}")


if __name__ == "__main__":
    main()
