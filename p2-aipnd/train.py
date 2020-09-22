import argparse
import os
import torch
from data_preparation import DataPreparation
from model import ImageModelWrapper


def train(model_wrapper, dp, epochs, gpu):
    """
    Trains the passed model with the desired parameters. Ensures automatic
    transfer of model to the GPU if available.
    After each training run model metrics are printed to standard out.

    Args:
        model_wrapper: ImageModelWrapper object containing model, criterion and optimizer used for model training
        dp: DataPreparation object containing data loaders for training and validation
        epochs: integer, amount of epochs to be run for training
        gpu: boolean, performs training on the gpu if true, on the cpu otherwise
    """

    device = torch.device("cuda" if gpu else "cpu")
    model = model_wrapper.model
    model.to(device)
    model.train()

    print(
        "\n--- starting training. Training results will be displayed after each epoch ---\n"
    )
    for epoch in range(epochs):
        running_loss = 0

        for images, labels in dp.dl_training:
            # runs training on passed model with pre-defined training dataset
            images, labels = images.to(device), labels.to(device)

            model_wrapper.optimizer.zero_grad()
            output = model.forward(images)
            loss = model_wrapper.criterion(output, labels)
            loss.backward()
            model_wrapper.optimizer.step()

            running_loss += loss.item()
        else:
            # runs validation after each epoch and prints out loss and
            # accuracy metrics
            model.eval()
            validation_loss = 0
            accuracy = 0

            with torch.no_grad():
                for (
                    images_validation,
                    labels_validation,
                ) in dp.dl_validation:
                    images_validation, labels_validation = (
                        images_validation.to(device),
                        labels_validation.to(device),
                    )

                    output = model.forward(images_validation)
                    loss = model_wrapper.criterion(output, labels_validation)
                    validation_loss += loss.item()

                    p = torch.exp(output)
                    top_p, top_class = p.topk(1, dim=1)
                    batch_accucary = top_class == labels_validation.view(
                        *top_class.shape
                    )
                    accuracy += torch.mean(
                        batch_accucary.type(torch.FloatTensor)
                    ).item()

            model.train()

            print(
                "Epoch: {}/{},\t Training Loss: {},\t Validation Loss: {},\t Validation Accuracy: {}".format(
                    epoch + 1,
                    epochs,
                    running_loss / len(dp.dl_training),
                    validation_loss / len(dp.dl_validation),
                    accuracy / len(dp.dl_validation),
                )
            )

    model.eval()
    model.to("cpu")


def main():
    """
    Checks arguments for validity and starts the training process according to
    the specified parameters.
    """

    parser = argparse.ArgumentParser(
        add_help=True,
        description="This file trains a new neural network on the given dataset.",
    )
    parser.add_argument(
        "data_dir",
        help="data directory containing data for training",
        action="store",
        type=check_dir_validity,
    )
    parser.add_argument(
        "--save_dir",
        action="store",
        default="./",
        dest="save_dir",
        help="directory to save model checkpoints. Expects full path, e.g. /path/to/dir/ without trailing '/'. By default it is stored in the current directory",
        type=check_dir_validity,
    )
    parser.add_argument(
        "--arch",
        action="store",
        choices=["vgg13", "vgg16"],
        default="vgg13",
        dest="arch",
        help="architecture to use as base for model training",
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.001,
        action="store",
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "--hidden_units",
        dest="hidden_units",
        type=int,
        default=512,
        action="store",
        help="amount of hidden units to use for classifier",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        dest="epochs",
        default=1,
        help="amount of training runs",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        dest="gpu",
        help="enables training on gpu to increase performance",
    )

    args = parser.parse_args()

    data_preparation = DataPreparation()
    data_preparation.prepare_training_data(args.data_dir)
    model_wrapper = ImageModelWrapper()
    model_wrapper.init_model(
        args.arch, int(args.hidden_units), float(args.learning_rate)
    )

    train(model_wrapper, data_preparation, int(args.epochs), args.gpu)

    model_wrapper.save(
        args.save_dir, int(args.epochs), data_preparation.class_to_idx
    )


def check_dir_validity(path):
    """
    Checks whether the given path is a valid path to a folder.

    Args:
        path: string, path to folder that is checked for existence

    Returns:
        string: absolute path of the folder

    Raises:
        NotADirectoryError: when the provided path is not a directory
    """

    p = os.path.abspath(os.path.realpath(os.path.expanduser(path)))

    if os.path.isdir(p):
        return p
    else:
        raise NotADirectoryError("Path: {} does not exist".format(p))


if __name__ == "__main__":
    main()
