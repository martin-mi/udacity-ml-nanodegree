import os
from data_preparation import DataPreparation
from model import ImageModelWrapper
import torch
import argparse
import json


def predict(image, model_wrapper, gpu, cat_to_name_path, topk):
    """
    Predict the class (or classes) of an image using a trained deep learning
    model.

    Args:
        image: Numpy array containing a processed image
        model_wrapper: ImageModelWrapper object containing model used for inference
        gpu: boolean, performs inference on the gpu if true, on the cpu otherwise
        cat_to_name_path: string, expects category to human-readable name dict in json format
        topk: integer, amount of classes to return
    """

    device = torch.device("cuda" if gpu else "cpu")
    image = image.unsqueeze(0).type(torch.FloatTensor)
    model = model_wrapper.model
    model.eval()

    output = model.to(device).forward(image.to(device))

    p = torch.exp(output)
    top_p, top_class = p.topk(topk, dim=1)

    class_list = create_class_name_mapping(
        model_wrapper.class_idx_mapping, cat_to_name_path, top_class
    )

    return top_p.squeeze(0).tolist(), class_list


def create_class_name_mapping(class_idx_mapping, cat_to_name_path, top_class):
    """
    Creates human-readble labels for predicted classes.

    Args:
        class_idx_mapping: list, contains class to index mapping
        cat_to_name_path: string, expects category to human-readable name dict in json format
        top_class: tensor, contains label classes

    Returns:
        list: with human-readable names for label classes. If class to name dictionary is empty, returns just indeces
    """

    idx_to_class = {v: k for (k, v) in class_idx_mapping.items()}

    cat_to_name = []
    with open(cat_to_name_path, "r") as f:
        cat_to_name = json.load(f)

    class_list = []
    for flower_class in top_class.squeeze(0).tolist():
        idx = idx_to_class[flower_class]

        if len(cat_to_name) == 0:
            class_list.append(idx)
        else:
            class_list.append(cat_to_name[idx])

    return class_list


def main():
    """
    Checks arguments for validity and starts the inference process according
    to the specified parameters.
    """

    parser = argparse.ArgumentParser(
        add_help=True,
        description="This file performs inference on the passed image and returns the probabilities for the inferred class.",
    )
    parser.add_argument(
        "image_path",
        help="path to image on which inference should be done",
        type=check_file_existence,
    )
    parser.add_argument(
        "checkpoint_path",
        help="path to checkpoint containing the model to be used for inference",
        type=check_file_existence,
    )
    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=1,
        action="store",
        help="amount of classes to return as result of this application",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        dest="gpu",
        help="enables inference on gpu to increase performance",
    )
    parser.add_argument(
        "--category_names",
        dest="category_names_path",
        type=check_file_existence,
        default="cat_to_name.json",
        action="store",
        help="path to file containing mapping of categories to real names",
    )

    args = parser.parse_args()

    data_preparation = DataPreparation()
    image = data_preparation.transform_image(args.image_path)

    model_wrapper = ImageModelWrapper()
    model_wrapper.load(args.checkpoint_path)

    top_p, class_list = predict(
        image,
        model_wrapper,
        args.gpu,
        args.category_names_path,
        int(args.top_k),
    )

    for p, name in zip(top_p, class_list):
        print("Flower is {} with probability {}%".format(name, p * 100))


def check_file_existence(path):
    """
    Checks whether the given path is a valid path to a file.

    Args:
        path: string, path to file that is checked for existence

    Returns:
        string: absolute path of the file

    Raises:
        Exception: when the provided path is not a file
    """

    p = os.path.abspath(os.path.realpath(os.path.expanduser(path)))
    if os.path.isfile(p):
        return p
    else:
        raise Exception(f"File: {p} does not exist or is not a file")


if __name__ == "__main__":
    main()
