from random import shuffle
import numpy as np
import torch
from torch.utils.data import Dataset


def sample_frames_uniformly(data, num_frames):
    """Uniformly samples data to a specified number of frames. """
    total_frames, num_joints, num_dims = data.shape
    if total_frames == num_frames:
        return data
    interval = total_frames / num_frames
    # Select frames at uniform intervals
    selected_indices = [int(i * interval) for i in range(num_frames)]
    return data[selected_indices]


def augment_data(data, class_labels, state_labels, positive_samples):
    """Applies data augmentation techniques like scaling and adding noise."""

    def apply_scaling(dataset):
        """Scales the dataset by a random factor within a specified range."""
        ratio = 0.1  # Scaling ratio
        for idx in range(dataset.shape[0]):
            factor = np.random.uniform(1 - ratio, 1 + ratio)
            # Apply scaling to x, y, z coordinates of each joint
            dataset[idx][:, :3] *= factor
        return dataset

    def add_noise(dataset):
        """Adds random noise to selected joints in the dataset."""
        ratio = 0.1  # Noise ratio
        num_joints = dataset.shape[2]
        all_joints = list(range(1, num_joints))
        shuffle(all_joints)
        selected_joints = all_joints[:5]  # Select 5 joints to add noise

        for idx in range(dataset.shape[0]):
            for joint_id in selected_joints:
                factor = np.random.uniform(1 - ratio, 1 + ratio)
                # Apply noise factor to each time frame for the selected joint
                dataset[idx][:, joint_id] *= factor
        return dataset

    # Create copies of the original data
    augmented_data = data.copy()
    augmented_class_labels = class_labels.copy()
    augmented_state_labels = state_labels.copy()
    augmented_positive_samples = positive_samples.copy()

    random_offset = np.random.randint(5)
    # Apply scaling augmentation to a subset of data
    augmented_data = np.append(augmented_data, apply_scaling(data[random_offset::5]), axis=0)
    augmented_class_labels = np.append(augmented_class_labels, class_labels[random_offset::5], axis=0)
    augmented_state_labels = np.append(augmented_state_labels, state_labels[random_offset::5], axis=0)
    augmented_positive_samples = np.append(augmented_positive_samples, apply_scaling(positive_samples[random_offset::5]), axis=0)

    return augmented_data, augmented_class_labels, augmented_state_labels, augmented_positive_samples


class SkeletonDataset(Dataset):
    """Custom Dataset for loading skeleton data, with optional data augmentation."""

    def __init__(self, data, class_labels, state_labels, positive_samples, mode="train", augment=False):
        if augment and mode == "train":
            data, class_labels, state_labels, positive_samples = augment_data(
                data, class_labels, state_labels, positive_samples)

        self.data = torch.tensor(data, dtype=torch.float32)
        self.class_labels = torch.tensor(class_labels, dtype=torch.float32)
        self.state_labels = torch.tensor(state_labels, dtype=torch.float32)
        self.positive_samples = torch.tensor(positive_samples, dtype=torch.float32)
        self.mode = mode

        # Ensure that data and labels have the same length
        assert len(self.data) == len(self.class_labels) == len(self.state_labels), \
            "Data and labels must have the same length after augmentation."

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, index):
        skeleton = self.data[index]
        class_label = int(self.class_labels[index])
        state_label = self.state_labels[index]
        positive_sample = self.positive_samples[index]

        return skeleton, class_label, state_label, positive_sample
