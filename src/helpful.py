import pandas as pd
import os
import torch
from PIL import Image
from functools import lru_cache


@lru_cache(maxsize=1)
def load_clip(clip_model_name="ViT-B-32-quickgelu", pretrained="laion400m_e32"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained
    )
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    return model, preprocess, tokenizer


@torch.no_grad()
def preprocess_image_paths(base_path: str, images: pd.Series, preprocess):
    """
    TODO add batch size to iterate over instaed of loading all into memory
    """

    preprocessed_images = (
        images.apply(lambda x: Image.open(os.path.join(base_path, x)))
        .apply(lambda x: preprocess(x).unsqueeze(0))
        .tolist()
    )

    # [n, 3, 224, 224] where n is the number of images
    # in this case, each image is 3x224x224 sized
    images_tensor = torch.cat(preprocessed_images, dim=0)

    return images_tensor


@torch.no_grad()
def preprocessed_to_clip_encodings(images_tensor: torch.Tensor, model):
    """
    TODO add batch size to iterate over instead of loading all into memory
    """
    # [n, 512] given each encoding is 512 dims
    image_features = model.encode_image(images_tensor)

    # then transform back into a df with n rows of 512 columns
    return image_features.detach().cpu().tolist()


def batches(tensor: torch.Tensor, batch_size: int):
    for i in range(0, len(tensor), batch_size):
        yield tensor[i : i + batch_size]


def batched_preprocessed_to_clip_encodings(
    images_tensor: torch.Tensor, model, batch_size=64
):
    image_features = []
    for batch in batches(images_tensor, batch_size):
        image_features.append(model.encode_image(batch).detach().cpu().tolist())
    return image_features
