import numpy as np
import torch
from torch import nn


class FeatureExtractor:
    def __init__(
            self,
            model: nn.Module,
            n_remove_layers: int,
            n_features,
            device
    ):
        self.model = model
        self.n_remove_layers = n_remove_layers
        self.n_features = n_features
        self.device = device

    def extract_features(self, dataset):
        model = self.prepare_model()
        image_codes = np.zeros((len(dataset), self.n_features))
        y_codes = np.zeros(len(dataset))
        for index, data in enumerate(dataset):
            image, label = data
            image_shape = image.shape
            image = image.to(self.device).view(1, *image_shape)
            with torch.no_grad():
                codes = model(image).view(1, self.n_features).to(self.device)
            image_codes[index] = codes.cpu().numpy()
            y_codes[index] = label
        return image_codes, y_codes

    def prepare_model(self):
        new_model = list(self.model.children())[:-self.n_remove_layers]
        return nn.Sequential(*new_model).to(self.device)
