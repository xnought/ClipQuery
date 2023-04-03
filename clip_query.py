from __future__ import annotations
import os
import torch
from PIL import Image
import open_clip
from tqdm import tqdm


def batch_size_iter(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def flatten(l, depth=1):
    flat = []

    def _flatten(l, depth):
        for item in l:
            if depth > 0 and isinstance(item, list):
                _flatten(item, depth - 1)
            else:
                flat.append(item)

    _flatten(l, depth)
    return flat


class ClipQuery:
    """ClipQuery
    Creates a clip model with methods on top to query it easily on dataframes
    Current prime usage: have one text and want to assign a score to each image to show how well they match
    For example: "a dog" and pictures of dogs hopefully have larger clip scores
    """

    def __init__(
        self,
        clip_model_name="ViT-B-32-quickgelu",
        pretrained="laion400m_e32",
        device="auto",
    ):
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained
        )
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

    @torch.no_grad()
    def encode_images(self, image_paths: list[str], base_path: str = "", batch_size=32):
        batches = []
        for batch in tqdm(
            batch_size_iter(image_paths, batch_size),
            total=len(image_paths) // batch_size,
        ):
            preprocessed_images: list[torch.Tensor] = [
                self.preprocess(Image.open(os.path.join(base_path, x))).unsqueeze(0)
                for x in batch
            ]
            # [batch, 3, 224, 224] where batch is the number of images in the batch
            # in this case, each image is 3x224x224 sized
            images_tensor = torch.cat(preprocessed_images, dim=0).to(self.device)
            image_features = self.model.encode_image(images_tensor)
            batches.append(image_features.detach().cpu().tolist())
        return flatten(batches, depth=1)

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def query(
        self,
        image_encodings: list[list[float]],
        text: str,
        normalize=False,
        batch_size=32,
    ):
        # encode the text in CLIP space
        tokens = self.tokenizer([text]).to(self.device)
        text_encoding = self.model.encode_text(tokens)
        batches = []
        for batch in tqdm(
            batch_size_iter(image_encodings, batch_size),
            total=len(image_encodings) // batch_size,
        ):
            # images are already encoded in CLIP space
            images_encoding = torch.tensor(batch, dtype=torch.float32).to(self.device)
            # compute scores for each image
            clip_scores = (
                images_encoding @ text_encoding.T
            )  # shapes: [n, vector_size] @ [vector_size, 1] = [n, 1]
            if normalize is True:
                clip_scores = clip_scores / torch.norm(clip_scores, dim=0, keepdim=True)
            batches.append(clip_scores.squeeze().detach().cpu().tolist())
        return flatten(batches, depth=1)

    def __repr__(self):
        return f"ClipQuery(device={self.device}, model={self.model})"
