from oir_to_tiff import OIRConverter
from pathlib import Path


def main():
    # Input and output paths
    input_path = Path(
        "data/input/793_102.18to101.68_6steps01_20_30_2k_1.5offset.oir")
    output_path = Path("data/output/converted.ome.tiff")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize converter
        converter = OIRConverter()

        # Convert file
        print(f"Converting {input_path.name} to OME-TIFF...")
        converter.convert(input_path, output_path)
        print(f"Successfully converted to {output_path.name}")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    main()
