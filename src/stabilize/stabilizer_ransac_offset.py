import numpy as np
import tifffile
import cv2
import logging


class RANSACStabilizer:
    def __init__(self, input_path, output_path):
        """
        Initialize the Image Stabilizer with input and output paths.

        Args:
            input_path (str or Path): Path to the input TIFF file.
            output_path (str or Path): Path where the stabilized TIFF will be saved.
        """
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.logger = logging.getLogger(__name__)

    def stabilize(self):
        """
        Perform image stabilization on the input TIFF file and save the result.
        """
        # Read the TIFF file
        self.logger.info(f"Reading TIFF file from {self.input_path}")
        with tifffile.TiffFile(self.input_path) as tif:
            # Assume the first series
            series = tif.series[0]
            axes = series.axes  # e.g., 'TZCYX'
            data = series.asarray()
            ome_metadata = tif.ome_metadata

        # Convert data to uint8 for OpenCV processing
        data_min = data.min()
        data_max = data.max()
        data_uint8 = ((data - data_min) / (data_max - data_min)
                      * 255).astype(np.uint8)

        # Find the axes indices
        t_index = axes.find('T')

        # Ensure that time axis exists
        if t_index == -1:
            raise ValueError("No time axis 'T' found in data")

        num_frames = data.shape[t_index]
        self.logger.info(f"Number of time frames: {num_frames}")

        # Prepare an array to hold the stabilized data
        stabilized_data = np.copy(data)

        # Initialize feature detector and matcher
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Use the first frame as the reference
        self.logger.info(
            "Using the first frame as reference for stabilization")
        ref_frame = self._get_frame(data_uint8, t_index, 0)
        ref_gray = self._convert_to_grayscale(ref_frame)

        ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

        for i in range(num_frames):
            self.logger.info(f"Processing frame {i+1}/{num_frames}")
            current_frame = self._get_frame(data_uint8, t_index, i)
            curr_gray = self._convert_to_grayscale(current_frame)

            curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

            if curr_des is not None and ref_des is not None:
                # Match descriptors
                matches = bf.match(ref_des, curr_des)
                matches = sorted(matches, key=lambda x: x.distance)

                # Extract matched keypoints
                ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches])
                curr_pts = np.float32(
                    [curr_kp[m.trainIdx].pt for m in matches])

                # Compute translation vector
                # Estimate transformation using RANSAC to reject outliers
        if len(matches) >= 3:
            # Extract matched keypoints
            ref_pts = np.float32(
                [ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_pts = np.float32(
                [curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate affine transformation with RANSAC
            matrix, inliers = cv2.estimateAffinePartial2D(
                curr_pts, ref_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=2000,
                confidence=0.99
            )

            if matrix is not None:
                # Extract translation components
                shift_x = matrix[0, 2]
                shift_y = matrix[1, 2]

                # Clamp the shifts if necessary
                max_shift = 20  # Adjust as needed
                shift_x = np.clip(shift_x, -max_shift, max_shift)
                shift_y = np.clip(shift_y, -max_shift, max_shift)

                # Create translation matrix
                translation_matrix = np.float32([[1, 0, shift_x],
                                                [0, 1, shift_y]])

                # Warp the current frame
                stabilized_frame = cv2.warpAffine(
                    current_frame,
                    translation_matrix,
                    (current_frame.shape[1], current_frame.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
            else:
                self.logger.warning(f"Transformation matrix not found for frame {
                                    i+1}, skipping stabilization")
                stabilized_frame = current_frame
        else:
            self.logger.warning(f"Not enough matches found for frame {
                                i+1}, skipping stabilization")
            stabilized_frame = current_frame

            # Convert stabilized frame back to original data type and scale
            stabilized_frame_float = stabilized_frame.astype(
                np.float32) / 255.0 * (data_max - data_min) + data_min
            stabilized_frame_original_dtype = stabilized_frame_float.astype(
                data.dtype)

            # Insert the stabilized frame back into the data
            self._set_frame(stabilized_data, t_index, i,
                            stabilized_frame_original_dtype)

        # Save the stabilized data
        self.logger.info(f"Saving stabilized data to {self.output_path}")
        with tifffile.TiffWriter(self.output_path, bigtiff=True, ome=True) as tif:
            tif.write(stabilized_data, metadata={
                      'axes': axes}, description=ome_metadata)
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
