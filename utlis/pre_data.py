import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from .skeleton import *


def prepare_dataset(file_pattern, frame_count, use_quaternions=True):
    data_segments = []
    labels = []
    states = []
    positive_samples = []
    null_state_value = 4  # Default state value when 'state' column is missing
    samples_by_class = {}  # Dictionary to store samples grouped by label
    n_joints = 11  # Adjust based on actual number of joints

    # Retrieve a sorted list of files matching the file pattern
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found matching pattern {file_pattern}")
        return [], [], [], []

    # Iterate over each file in the list
    for filepath in tqdm(file_list):
        try:
            # Load data from the CSV file
            data_frame = load_csv_data(filepath)
            data_length, num_columns = data_frame.shape

            # Skip files that are too short
            if data_length < frame_count:
                print(f"Data length ({data_length}) is less than frame count ({frame_count}) in file {filepath}. Skipping.")
                continue

            # Adjust data length to be a multiple of frame_count
            frames_to_discard = data_length % frame_count
            if frames_to_discard != 0:
                data_frame = data_frame.iloc[frames_to_discard:]

            # Split the data into segments of length 'frame_count'
            num_segments = len(data_frame) // frame_count
            data_splits = np.array_split(data_frame, num_segments)

            for segment in data_splits:
                # Extract the label from the filename
                label = get_label_from_filename(filepath)
                if label is None:
                    continue  # Skip if label is unrecognized

                # Check if 'state' column exists and handle accordingly
                if 'state' in segment.columns:
                    # Extract 'state' values and remove the column from the segment
                    states.append(segment['state'].to_numpy())
                    segment = segment.drop('state', axis=1)
                else:
                    # Assign default state values if 'state' column is missing
                    states.append(null_state_value * np.ones(len(segment)))

                # Check if number of columns is divisible by n_joints
                num_columns = segment.shape[1]
                if num_columns % n_joints != 0:
                    print(f"Number of columns {num_columns} cannot be evenly divided by {n_joints} in file {filepath}.")
                    continue  # Skip this segment

                # Reshape segment data into (frames, joints, features)
                segment_array = np.array(np.split(segment.to_numpy(), n_joints, axis=1))
                segment_array = np.swapaxes(segment_array, 0, 1)
                data_segments.append(segment_array)
                labels.append(label)

                # Group samples by label for positive sampling
                if label not in samples_by_class:
                    samples_by_class[label] = []
                samples_by_class[label].append(segment_array)

                # Select a positive sample (previous sample of the same class if available)
                if len(samples_by_class[label]) > 1:
                    positive_samples.append(samples_by_class[label][-2])
                else:
                    positive_samples.append(segment_array)

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    # Check if any data segments were collected
    if not data_segments:
        print("No valid data segments found.")
        return [], [], [], []

    # Convert lists to numpy arrays
    data_array = np.stack(data_segments)
    label_array = np.array(labels)
    state_array = np.stack(states)
    positive_samples_array = np.stack(positive_samples)

    # Replace any NaN values in state_array with the null_state_value
    state_array = np.nan_to_num(state_array, nan=null_state_value)
    return data_array, label_array, state_array, positive_samples_array


def load_csv_data(filepath):
    data = pd.read_csv(filepath)
    # Select columns starting from index 8; adjust this based on the actual data structure
    data = data.iloc[:, 8:]
    return data


def select_frames(data_frame, frame_count):
    total_frames, num_features = data_frame.shape
    if total_frames == frame_count:
        return data_frame
    interval = total_frames / frame_count
    # Calculate indices to select frames at uniform intervals
    selected_indices = [int(i * interval) for i in range(frame_count)]
    return data_frame.iloc[selected_indices]


def get_label_from_filename(filepath):
    if 'fist' in filepath:
        return 0
    elif 'cube' in filepath:
        return 1
    elif 'vertical' in filepath:
        return 2
    elif 'ring' in filepath:
        return 3
    elif 'pinky' in filepath:
        return 4
    elif 'scissors' in filepath:
        return 5
    elif 'slide' in filepath:
        return 6
    elif 'null' in filepath:
        return 7
    else:
        # Print a warning for unrecognized gestures
        print(f"Unrecognized gesture in filepath: {filepath}")
        return None
