###################################################################################################
#
# Rock-Paper-Scissors Dataset Loader
#
# Adapted from Maxim Integrated’s Cats vs Dogs example
#
###################################################################################################

import os
import torch
import torchvision
from torchvision import transforms
import ai8x

from PIL import Image
import errno
import shutil
import sys
import numpy

torch.manual_seed(0)

# ----------------------------- AUGMENTATION FUNCTIONS -----------------------------
def augment_affine_jitter_blur(orig_img):
    """Apply multiple transformations for augmentation"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=8),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.CenterCrop((180, 180)),
        transforms.ColorJitter(brightness=.6, contrast=.6, saturation=.6),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
    ])
    return train_transform(orig_img)


def augment_blur(orig_img):
    """Simple center crop and blur"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((220, 220)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
    ])
    return train_transform(orig_img)


# ----------------------------- MAIN DATASET FUNCTION -----------------------------
def rps_get_datasets(data, load_train=True, load_test=True, aug=2):
    """
    Load Rock-Paper-Scissors dataset.
    Expected structure:
      data/rock_paper_scissors/train/rock/
      data/rock_paper_scissors/train/paper/
      data/rock_paper_scissors/train/scissors/
      data/rock_paper_scissors/test/rock/
      data/rock_paper_scissors/test/paper/
      data/rock_paper_scissors/test/scissors/
    """
    (data_dir, args) = data
    dataset_path = os.path.join(data_dir, "rock_paper_scissors")

    if not os.path.isdir(dataset_path):
        print("******************************************")
        print("Rock-Paper-Scissors dataset not found!")
        print("Please download it from:")
        print("https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors")
        print("Unzip into: 'data/rock_paper_scissors'")
        print("Folder structure should be:")
        print("  data/rock_paper_scissors/train/[rock|paper|scissors]")
        print("  data/rock_paper_scissors/test/[rock|paper|scissors]")
        print("******************************************")
        sys.exit("Dataset not found!")

    processed_dataset_path = os.path.join(dataset_path, "augmented")

    if os.path.isdir(processed_dataset_path):
        print("Augmented folder exists. Remove it to regenerate.")
    else:
        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")
        processed_train_path = os.path.join(processed_dataset_path, "train")
        processed_test_path = os.path.join(processed_dataset_path, "test")

        os.makedirs(processed_train_path, exist_ok=True)
        os.makedirs(processed_test_path, exist_ok=True)

        # create label folders
        for d in os.listdir(train_path):
            os.makedirs(os.path.join(processed_train_path, d), exist_ok=True)
        for d in os.listdir(test_path):
            os.makedirs(os.path.join(processed_test_path, d), exist_ok=True)

        # copy test data
        test_cnt = 0
        for (dirpath, _, filenames) in os.walk(test_path):
            print(f'Copying {dirpath} -> {processed_test_path}')
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.png')):
                    relsourcepath = os.path.relpath(dirpath, test_path)
                    destpath = os.path.join(processed_test_path, relsourcepath)
                    shutil.copyfile(os.path.join(dirpath, filename), os.path.join(destpath, filename))
                    test_cnt += 1

        # copy and augment train data
        train_cnt = 0
        for (dirpath, _, filenames) in os.walk(train_path):
            print(f'Copying and augmenting {dirpath} -> {processed_train_path}')
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.png')):
                    relsourcepath = os.path.relpath(dirpath, train_path)
                    destpath = os.path.join(processed_train_path, relsourcepath)
                    srcfile = os.path.join(dirpath, filename)
                    destfile = os.path.join(destpath, filename)

                    # copy original
                    shutil.copyfile(srcfile, destfile)
                    train_cnt += 1

                    orig_img = Image.open(srcfile).convert("RGB")

                    # blur augmentation
                    aug_img = augment_blur(orig_img)
                    augfile = destfile[:-4] + '_ab0.jpg'
                    aug_img.save(augfile)
                    train_cnt += 1

                    # jitter & affine augmentations
                    for i in range(aug):
                        aug_img = augment_affine_jitter_blur(orig_img)
                        augfile = destfile[:-4] + f'_aj{i}.jpg'
                        aug_img.save(augfile)
                        train_cnt += 1

        print(f'Augmented dataset: {test_cnt} test images, {train_cnt} train images.')

    # Paths
    processed_train_path = os.path.join(processed_dataset_path, "train")
    processed_test_path = os.path.join(processed_dataset_path, "test")

    # train transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ai8x.normalize(args=args)
    ])

    # test transforms
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ai8x.normalize(args=args)
    ])

    # datasets
    train_dataset = torchvision.datasets.ImageFolder(root=processed_train_path,
                                                     transform=train_transform) if load_train else None

    test_dataset = torchvision.datasets.ImageFolder(root=processed_test_path,
                                                    transform=test_transform) if load_test else None

    return train_dataset, test_dataset


# ----------------------------- DATASET REGISTRATION -----------------------------
datasets = [
    {
        'name': 'rock_paper_scissors',
        'input': (3, 128, 128),
        'output': ('rock', 'paper', 'scissors'),
        'loader': rps_get_datasets,
    },
]
