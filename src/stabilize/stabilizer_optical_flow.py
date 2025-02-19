import numpy as np
import tifffile
import cv2
import logging
from pathlib import Path


class ImageStabilizer:
    def __init__(self, method="optical_flow"):
        """
        Initialize the Image Stabilizer.

        Args:
            method (str): Stabilization method to use (default: "optical_flow")
        """
        self.method = method
        self.logger = logging.getLogger(__name__)

    def stabilize(self, data, axes=None):
        """
        Perform image stabilization on the input data.

        Args:
            data (numpy.ndarray): Input image data
            axes (str, optional): Axes description (e.g., 'TZCYX')

        Returns:
            numpy.ndarray: Stabilized image data
        """
        self.logger.info("Starting stabilization")

        # If axes not provided, assume first dimension is time
        if axes is None:
            t_index = 0
        else:
            t_index = axes.find('T')
            if t_index == -1:
                raise ValueError("No time axis 'T' found in data")

        # Convert data to uint8 for OpenCV processing
        data_min = data.min()
        data_max = data.max()
        data_uint8 = ((data - data_min) / (data_max - data_min)
                      * 255).astype(np.uint8)

        num_frames = data.shape[t_index]
        self.logger.info(f"Number of time frames: {num_frames}")

        # Prepare an array to hold the stabilized data
        stabilized_data = np.copy(data)

        # Use the first frame as the reference
        self.logger.info(
            "Using the first frame as reference for stabilization")
        ref_frame = self._get_frame(data_uint8, t_index, 0)
        ref_gray = self._convert_to_grayscale(ref_frame)

        # Detect good features to track
        p0 = cv2.goodFeaturesToTrack(
            ref_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7,
        )

        # Initialize cumulative shifts
        cumulative_shift_x = 0
        cumulative_shift_y = 0

        for i in range(num_frames):
            self.logger.info(f"Processing frame {i+1}/{num_frames}")
            current_frame = self._get_frame(data_uint8, t_index, i)
            curr_gray = self._convert_to_grayscale(current_frame)

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                ref_gray, curr_gray, p0, None)

            if p1 is not None and st.sum() >= 10:
                # Select good points
                good_old = p0[st == 1]
                good_new = p1[st == 1]

                # Compute displacement
                displacements = good_new - good_old
                median_dx = np.median(displacements[:, 0])
                median_dy = np.median(displacements[:, 1])

                # Clamp the shifts
                max_shift = 20  # Adjust as needed
                shift_x = np.clip(median_dx, -max_shift, max_shift)
                shift_y = np.clip(median_dy, -max_shift, max_shift)

                # Accumulate shifts
                cumulative_shift_x += shift_x
                cumulative_shift_y += shift_y

                # Create translation matrix
                translation_matrix = np.float32([
                    [1, 0, -cumulative_shift_x],
                    [0, 1, -cumulative_shift_y],
                ])

                # Warp the current frame
                stabilized_frame = cv2.warpAffine(
                    current_frame,
                    translation_matrix,
                    (current_frame.shape[1], current_frame.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )

                # Update reference frame and points
                ref_gray = curr_gray.copy()
                p0 = cv2.goodFeaturesToTrack(
                    ref_gray,
                    maxCorners=200,
                    qualityLevel=0.01,
                    minDistance=30,
                    blockSize=7,
                )
            else:
                self.logger.warning(f"Not enough points to track for frame {
                                    i+1}, skipping stabilization")
                stabilized_frame = current_frame
                # Do not update cumulative shifts

            # Convert stabilized frame back to original data type and scale
            stabilized_frame_float = stabilized_frame.astype(
                np.float32) / 255.0 * (data_max - data_min) + data_min
            stabilized_frame_original_dtype = stabilized_frame_float.astype(
                data.dtype)

            # Insert the stabilized frame back into the data
            self._set_frame(stabilized_data, t_index, i,
                            stabilized_frame_original_dtype)

        return stabilized_data

    def stabilize_file(self, input_path, output_path):
        """
        Stabilize a TIFF file and save the result.

        Args:
            input_path (str or Path): Path to input TIFF file
            output_path (str or Path): Path to save stabilized TIFF
        """
        self.logger.info(f"Reading TIFF file from {input_path}")
        print(f"Starting stabilization of {input_path} to {output_path}")

        try:
            with tifffile.TiffFile(str(input_path)) as tif:
                series = tif.series[0]
                axes = series.axes
                data = series.asarray()
                ome_metadata = tif.ome_metadata
                print(
                    f"Successfully read input file. Shape: {data.shape}, Axes: {axes}")

            stabilized_data = self.stabilize(data, axes)
            print(
                f"Stabilization complete. Output shape: {stabilized_data.shape}")

            self.logger.info(f"Saving stabilized data to {output_path}")
            print(f"Attempting to save to {output_path}")

            with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
                tif.write(stabilized_data, metadata={
                          'axes': axes}, description=ome_metadata)

            print(f"Save complete. File exists: {Path(output_path).exists()}")
            print(f"File size: {Path(output_path).stat().st_size}")

        except Exception as e:
            print(f"Error during stabilization: {str(e)}")
            raise

        self.logger.info("Stabilization complete")

    def _get_frame(self, data, t_index, frame_number):
        slicing = [slice(None)] * data.ndim
        slicing[t_index] = frame_number
        frame = data[tuple(slicing)]
        # If frame has more than 2 dimensions, flatten it (e.g., max over Z and sum over channels)
        while frame.ndim > 2:
            frame = frame.max(axis=0)
        return frame

    def _convert_to_grayscale(self, frame):
        # Assuming frame is already in grayscale
        if frame.ndim == 2:
            return frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported frame format")

    def _set_frame(self, data, t_index, frame_number, frame_data):
        slicing = [slice(None)] * data.ndim
        slicing[t_index] = frame_number
        data[tuple(slicing)] = frame_data
