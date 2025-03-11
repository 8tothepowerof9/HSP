import os
import re
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class HandSignDataset(Dataset):
    def __init__(self, split="train", dir="data", final_processor=None):
        """An ASL Alphabet dataset. Each data points is represented by 21 landmarks with 3 dimensions: x, y, z
        The dataset will automatically extract angles, as well as apply translation and scale normalization on the landmarks.

        Args:
            split (str, optional): Split the dataset into train and test set. Defaults to "train".
            dir (str, optional): Path to dataset folder. Defaults to "data".
            save_processed (bool, optional): Save the processed dataset to a folder. Defaults to False.
            final_processor (callable, optional): A function that processes the processed landmarks. Useful if wants to create a graph from a data. Defaults to None.
        """
        self.split = split
        self.dir = dir
        self.final_processor = final_processor

        # Load the dataset from folder

        # If save_processed is True, save the processed dataset to a folder

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def __translation_norm__(self, landmarks, ref_point_idx):
        """Translate the hand landmarks so that the reference point is at the origin.

        Args:
            landmarks (array): NumPy array of shape (N, 3) representing hand landmarks
            ref_point_idx (int): _description_Index of the landmark to use as the new origin
        """
        reference_point = landmarks[ref_point_idx]
        translated_landmarks = landmarks - reference_point
        return translated_landmarks

    def __scale_norm__(self, landmarks, max_dist=1):
        """Scale the hand landmarks so that the maximum distance between any two landmarks is equal to desired_max_distance.

        Args:
            landmarks (array): NumPy array of shape (N, 3) representing hand landmarks
            max_dist (int): The desired maximum distance between any two landmarks. Default value is 1.
        """
        # Compute all pairwise distances
        distances = np.linalg.norm(
            landmarks[:, np.newaxis] - landmarks[np.newaxis, :], axis=2
        )

        # Find the maximum distance in the current set of landmarks
        current_max_dist = distances.max()

        # Calculate the scale factor
        scale_factor = max_dist / current_max_dist

        # Scale the landmarks
        scaled_landmarks = landmarks * scale_factor

        return scaled_landmarks

    def __get_label_mapping__(self):
        return {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9,
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "O": 14,
            "P": 15,
            "Q": 16,
            "R": 17,
            "S": 18,
            "T": 19,
            "U": 20,
            "V": 21,
            "W": 22,
            "X": 23,
            "Y": 24,
            "Z": 25,
            "1": 26,
            "2": 27,
            "3": 28,
            "4": 29,
            "5": 30,
            "6": 31,
            "7": 32,
            "8": 33,
            "9": 34,
            "0": 35,
        }
