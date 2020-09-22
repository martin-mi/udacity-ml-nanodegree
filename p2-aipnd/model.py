import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models


class ImageModelWrapper:
    """
    Wraps information related to models and hyperparameters anf offer utility
    functions like e.g. model saving and loading.
    """

    def init_model(self, arch, hidden_units, learning_rate):
        """
        Instantiates a pre-trained model from torchvision with a custom
        classifier that can be fitted to a new use case - flower data set in
        this case. Moreover, it initializes a criterion and optimizer used for
        loss calculation and backpropagation.

        Args:
            arch: string, specifies which pre-trained model to use.
            hidden_units: integer, size of the hidden layer of the custom classifier
            learning_rate: float, learning rate that gets passed to the optimizer
        """

        if hasattr(models, arch):
            self.model = getattr(models, arch)(pretrained=True)
        else:
            sys.exit(
                "Architecture '{}' not found. Please refer to the documentation for possible architectures at https://pytorch.org/docs/stable/torchvision/models.html".format(
                    arch
                )
            )

        self.model_type = arch

        for param in self.model.parameters():
            param.requires_grad = False

        self.input_size, self.hidden_units, self.output_size = (
            self.model.classifier[0].in_features,
            hidden_units,
            102,
        )
        self.model.classifier = ImageClassifier(
            self.input_size, hidden_units, self.output_size
        )

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=learning_rate
        )

    def save(self, path, epochs, class_idx_mapping):
        """
        Saves the model on the file system including all relevant information
        needed to rebuild the model. It contains amongst others layer sizes of
        the custom classifier, the trained weights, optimizer and criterion
        information and also class mappings to retrieve a human-readable
        representation of the predicted object.

        Args:
            path: string, path where the model is saved with name 'checkpoint.pth'
            epochs: integer, specifies how many epochs were used to train the model
            class_idx_mapping: dictionary, mapping of classes to indices
        """

        self.epochs = epochs
        self.class_idx_mapping = class_idx_mapping

        checkpoint = {
            "input_size": self.input_size,
            "hidden_layer_size": self.hidden_units,
            "output_size": self.output_size,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion": self.criterion,
            "class_idx_mapping": class_idx_mapping,
            "epochs": epochs,
            "model_type": self.model_type,
        }

        model_path = path + "/checkpoint.pth"
        torch.save(checkpoint, model_path)
        print("--- model stored successfully at {}".format(model_path))

    def load(self, path):
        """
        Loads model from the specified path on the file system and initializes
        member variables contaning relevant model properties, e.g. class to
        index mapping, criterion and optimizer state dict.

        Args:
            path: string, full path where the model is saved
        """

        checkpoint = torch.load("/" + path)

        if hasattr(models, checkpoint["model_type"]):
            self.model = getattr(models, checkpoint["model_type"])()
        else:
            sys.exit(
                "Error while loading saved model - architecture '{}' not found.".format(
                    arch
                )
            )

        self.model.classifier = ImageClassifier(
            checkpoint["input_size"],
            checkpoint["hidden_layer_size"],
            checkpoint["output_size"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])

        # register additional parameters in case model is used for further training
        self.epochs = checkpoint["epochs"]
        self.class_idx_mapping = checkpoint["class_idx_mapping"]
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        self.criterion = checkpoint["criterion"]

        print("--- model loaded successfully from {}".format(path))


class ImageClassifier(nn.Module):
    """
    Custom classifier used to train the last layer of the model to the new
    dataset.

    Args:
        input_size: integer, size of input layer
        hidden_layer_size: integer, size of hidden layer
        output_size: integer, size of output layer
    """

    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x
