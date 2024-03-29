from typing import Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class TripletSVHN(Dataset):
    """
    Based on https://github.com/adambielski/siamese-triplet

    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            indices_train: np.ndarray,
            indices_test: np.ndarray,
            transform: torchvision.transforms.Compose,
            phase: list,
            seed: int,
            return_labels: bool = False
    ) -> None:
        self.dataset = dataset
        self.indices_train = indices_train
        self.indices_test = indices_test
        self.transform = transform
        self.phase = phase
        self.seed = seed
        self.return_labels = return_labels

        if self.phase == 'train':
            self.train_labels = self.dataset.labels[self.indices_train]
            self.train_data = self.dataset.data[self.indices_train]
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.labels[self.indices_test]
            self.test_data = self.dataset.data[self.indices_test]
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(self.seed)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(
            self,
            index: int
    ) -> Tuple[torch.Tensor]:
        if self.phase == 'train':
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(np.moveaxis(img1, 0, 2))
        img2 = Image.fromarray(np.moveaxis(img2, 0, 2))
        img3 = Image.fromarray(np.moveaxis(img3, 0, 2))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        if self.return_labels:
            return (img1, img2, img3), label1
        else:
            return (img1, img2, img3), []

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_labels)
        else:
            return len(self.test_labels)
