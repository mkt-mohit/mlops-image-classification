"""
src/data/dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset for the preprocessed Cats vs Dogs images.
Training split gets augmentation; val/test get only normalisation.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── ImageNet stats (standard for CNNs pre-trained on ImageNet) ────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Return torchvision transforms for a given *split*.

    Training  → RandomHorizontalFlip + ColorJitter + RandomRotation + Normalize
    Val/Test  → CenterCrop + Normalize only
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        # images are already 224×224 from preprocessing
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


class CatsDogsDataset(Dataset):
    """
    Folder layout expected:
        processed/{split}/cats/*.jpg
        processed/{split}/dogs/*.jpg

    Labels:  cats → 0 | dogs → 1
    """

    CLASS_TO_IDX = {"cats": 0, "dogs": 1}

    def __init__(self, processed_dir: Path, split: str, img_size: int = 224):
        self.split     = split
        self.transform = get_transforms(split, img_size)
        self.samples: list[tuple[Path, int]] = []

        split_dir = processed_dir / split
        for class_name, label in self.CLASS_TO_IDX.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {class_dir}\n"
                    "Run src/data/preprocess.py first."
                )
            for img_path in sorted(class_dir.iterdir()):
                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


def build_dataloaders(
    processed_dir: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
) -> dict[str, DataLoader]:
    """
    Returns a dict with 'train', 'val', 'test' DataLoaders.
    """
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = CatsDogsDataset(processed_dir, split, img_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders