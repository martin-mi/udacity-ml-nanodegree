from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image
import numpy as np
import torch


class DataPreparation:
    """
    Offers helper methods for data preprocessing that can be used for training
    and inference.
    """

    def prepare_training_data(self, data_dir):
        """
        Loads data for training, validation and testing. Data is loaded,
        normalized (according to the model properties of the pre-trained ImageNet models)
        and DataLoader member variables initialized that can be used for train/test purposes.

        Args:
            data_dir: string, folder with data. Expects subfolders named 'train', 'valid' and 'test' for training, validation and testing
        """

        train_dir = data_dir + "/train"
        valid_dir = data_dir + "/valid"
        test_dir = data_dir + "/test"

        transforms_training = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        transforms_testing = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset_training = datasets.ImageFolder(
            train_dir, transform=transforms_training
        )
        dataset_validation = datasets.ImageFolder(
            valid_dir, transform=transforms_testing
        )
        dataset_testing = datasets.ImageFolder(
            test_dir, transform=transforms_testing
        )

        self.class_to_idx = dataset_training.class_to_idx

        self.dl_training = data.DataLoader(
            dataset_training, batch_size=64, shuffle=True
        )
        self.dl_validation = data.DataLoader(dataset_validation, batch_size=64)
        self.dl_testing = data.DataLoader(dataset_testing, batch_size=64)

    def transform_image(self, path):
        """
        Scales, crops, and normalizes image at given path for processing
        via PyTorch.

        Args:
            path: string, path to the image

        Returns:
            Numpy array containing the processed image
        """

        image = Image.open(path)
        image.thumbnail((256, 256))

        new_size = 224
        left = (image.size[0] - new_size) / 2
        upper = (image.size[1] - new_size) / 2
        right = (image.size[0] + new_size) / 2
        lower = (image.size[1] + new_size) / 2

        cropped = image.crop((left, upper, right, lower))

        # adjust color channel features
        np_image = np.array(cropped) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        np_image = (np_image - mean) / std
        np_image = np_image.transpose(2, 0, 1)

        return torch.from_numpy(np_image)
