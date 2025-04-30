
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
import random
import os
import numpy as np

# Try importing sklearn, handle potential ImportError
try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# List of supported image datasets. "mini_mnist" is a stratified 5‑k sample of the
# standard MNIST training split (introduced for faster experimentation).
IMAGE_DATASETS = [
    "mnist",
    "mini_mnist",  # new – 5 000 examples (500 per class) uniformly sampled from MNIST
    "mini_mnist_8chanel",  # 5 000 examples, each duplicated across 8 channels (28×28)
    "mnist_flat",
    "cifar10",
    "cifar100",
    "svhn",
    "imagenet",
    "mini_imagenet",  # Mini-ImageNet subset stored as class folders under data/mini_imagenet
]
REGRESSION_DATASETS = ["ones", "california_housing"]
NLP_DATASETS = ["imdb"]

def generate_imagedata(dataset_name):
    # Define common transformations including resizing
    resize_transform = transforms.Resize((32, 32))
    to_tensor_transform = transforms.ToTensor()
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "mini_mnist":
        """Return a stratified subset (500 samples per class, total 5 000) of the
        MNIST training split. The transform is identical to the full MNIST case
        so models can be swapped without further changes.
        """

        import random

        # Keep identical preprocessing as for the regular MNIST variant so that
        # downstream architectures receive 3×32×32 tensors.
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5,), (0.5,)),
        ])

        full_train_set = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        # Build stratified indices: 500 examples for each of the 10 classes.
        targets = full_train_set.targets  # Tensor of shape (60000,)
        subset_indices = []
        samples_per_class = 500

        for cls in range(10):
            cls_indices = (targets == cls).nonzero(as_tuple=False).flatten().tolist()
            # Ensure reproducibility across runs by fixing the RNG seed here if the
            # caller has *not* already set it. We opt for deterministic sampling
            # relying on the global random module state.
            if len(cls_indices) < samples_per_class:
                raise ValueError(
                    f"Requested {samples_per_class} samples for class {cls}, "
                    f"but only {len(cls_indices)} are available."
                )
            subset_indices.extend(random.sample(cls_indices, samples_per_class))

        # Shuffle the combined indices so that subsequent DataLoader shuffling is
        # optional (helps when shuffle=False is used for debugging).
        random.shuffle(subset_indices)

        # Create a subset dataset
        mini_train_set = torch.utils.data.Subset(full_train_set, subset_indices)

        # Expose `.data` and `.targets` attributes (needed by helper utilities
        # such as `get_dataset_dimensionality`).
        mini_train_set.data = full_train_set.data[subset_indices]
        mini_train_set.targets = targets[subset_indices]

        return mini_train_set

    elif dataset_name == "mini_mnist_8chanel":
        """Stratified 5 k subset of MNIST with each image expanded to 8 channels.

        The underlying grayscale image is replicated across 8 identical
        feature maps, producing tensors of shape (8, 28, 28).
        """

        import random

        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # produces (1, H, W)
                transforms.Lambda(lambda x: x.repeat(8, 1, 1)),  # (8, H, W)
                transforms.Normalize((0.5,) * 8, (0.5,) * 8),
            ]
        )

        full_train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

        targets = full_train_set.targets  # torch.Tensor of length 60 000
        subset_indices = []
        samples_per_class = 500

        for cls in range(10):
            cls_idx = (targets == cls).nonzero(as_tuple=False).flatten().tolist()
            subset_indices.extend(random.sample(cls_idx, samples_per_class))

        random.shuffle(subset_indices)

        mini_train_set = torch.utils.data.Subset(full_train_set, subset_indices)
        mini_train_set.data = full_train_set.data[subset_indices]
        mini_train_set.targets = targets[subset_indices]

        return mini_train_set

    elif dataset_name == "mnist_flat":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten the image after normalization
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_set.data = train_set.data.view(-1,28*28)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "svhn":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "imagenet":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_set = ImageFolder(root='data/imagenet_data/data/imagenet', transform=transform)
        return train_set  # Return the dataset directly

    elif dataset_name == "mini_imagenet":
        """Return the Mini‑ImageNet training split.

        Expects the images arranged in ImageNet‑style folder hierarchy at
        ``data/mini_imagenet/train/<class_name>/*.jpg``.  If your copy is
        stored elsewhere, create a symlink or change the *root* path below.
        """

        # Mini‑ImageNet images are 84×84; we upsample to 224×224 for
        # compatibility with standard ImageNet models, or to 32×32 if you want
        # faster training.  Here we choose 64×64 to keep resolution moderate.
        minimg_resize = transforms.Resize((64, 64))

        transform = transforms.Compose(
            [
                minimg_resize,
                to_tensor_transform,
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        mini_root = "data/mini_imagenet/train"
        # Basic sanity‑check to help new users: if the directory is missing or
        # empty we raise a descriptive error explaining how to obtain the
        # dataset.
        import os

        if not os.path.isdir(mini_root) or len(os.listdir(mini_root)) == 0:

            raise FileNotFoundError(
                "Mini‑ImageNet dataset not found. Expected class folders under "
                f"'{mini_root}'. Please download the dataset (e.g. from "
                "https://github.com/yaoyao-liu/mini-imagenet-tools or Kaggle) "
                "and place/extract it so that each class has its own directory "
                "containing images."
            )

        train_set = ImageFolder(root=mini_root, transform=transform)
        return train_set
    
    else:
        print("Unknown dataset name")
        raise ValueError("Unknown dataset name")
    


# def generate_nlp_data(dataset_name, batch_size=32, max_vocab_size=10000, max_length=512):
#     if dataset_name.lower() != "imdb":
#         raise ValueError("Currently, only the 'IMDB' dataset is supported.")

#     # Load the IMDB dataset
#     train_iter, test_iter = IMDB()

#     # Tokenization
#     tokenizer = get_tokenizer('basic_english')

#     # Build vocabulary
#     def yield_tokens(data_iter):
#         for _, text in data_iter:
#             yield tokenizer(text)

#     vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
#     vocab.set_default_index(vocab["<unk>"])  # Set default index for unknown words

#     # Limit the size of the vocabulary
#     if len(vocab) > max_vocab_size:
#         vocab = vocab.freq_cutoff(max_vocab_size - 2)  # Adjust for special tokens

#     # Numericalize and pad text function
#     def process_text(text):
#         tokenized_text = tokenizer(text)
#         numericalized_text = vocab(tokenized_text)
#         padded_text = numericalized_text[:max_length] + [vocab['<pad>']] * (max_length - len(numericalized_text))
#         return torch.tensor(padded_text)

#     # Process the datasets
#     def collate_batch(batch):
#         label_list, text_list = [], []
#         for label, text in batch:
#             label_list.append(1 if label == 'pos' else 0)
#             processed_text = process_text(text)
#             text_list.append(processed_text)
#         return torch.tensor(label_list), torch.stack(text_list)

#     # Convert to DataLoader
#     train_dataset = to_map_style_dataset(train_iter)
#     test_dataset = to_map_style_dataset(test_iter)

#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

#     return train_dataloader, test_dataloader

def generate_regressiondata(dataset_name, n_samples=1000, d1=10, d2=1, scale_features=True):
    """
    Generates a synthetic or real regression dataset.

    Supported datasets:
      - "ones": Synthetic dataset where inputs and outputs are tensors of ones.
                 Uses n_samples, d1, d2 arguments.
      - "california_housing": The California Housing dataset from scikit-learn.
                              Ignores n_samples, d1, d2. Requires scikit-learn.

    Parameters:
        dataset_name (str): Name of the regression dataset. Must be in REGRESSION_DATASETS.
        n_samples (int): Number of samples for the "ones" dataset (default is 1000). Ignored otherwise.
        d1 (int): Dimension of the input features for the "ones" dataset (default is 10). Ignored otherwise.
        d2 (int): Dimension of the output for the "ones" dataset (default is 1). Ignored otherwise.
        scale_features (bool): If True and using "california_housing", scale features using StandardScaler.
                               (default is False).

    Returns:
        torch.utils.data.TensorDataset: A TensorDataset where each sample is a tuple (input, target).
                                        Input and target shapes depend on the dataset.

    Raises:
        ValueError: If the dataset_name is not recognized or dependencies are missing.
        ImportError: If 'california_housing' is requested but scikit-learn is not installed.
    """
    if dataset_name == "ones":
        # Create synthetic input and output tensors filled with ones
        X = torch.ones(n_samples, d1, dtype=torch.float32)
        y = torch.ones(n_samples, d2, dtype=torch.float32)
        # Create a TensorDataset from the generated tensors
        dataset = TensorDataset(X, y)
        return dataset

    elif dataset_name == "california_housing":
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required to load the California Housing dataset. "
                              "Please install it (`pip install scikit-learn`).")

        # Fetch the dataset using sklearn
        california_data = fetch_california_housing(as_frame=False) # Get NumPy arrays
        X_np = california_data.data
        y_np = california_data.target

        # Optionally scale features
        if scale_features:
            # Ensure StandardScaler is imported or available
            try:
                 from sklearn.preprocessing import StandardScaler
            except ImportError:
                 raise ImportError("StandardScaler requires scikit-learn.")
            scaler = StandardScaler()
            X_np = scaler.fit_transform(X_np)

        # Convert numpy arrays to torch tensors
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

        # --- Correction Here ---
        # Create and return a TensorDataset instead of the Bunch object
        dataset = TensorDataset(X_tensor, y_tensor)
        dataset.data = X_tensor
        dataset.targets = y_tensor
        # You could optionally attach metadata if needed elsewhere, but it's not standard for Dataset
        # dataset.feature_names = california_data.feature_names
        return dataset # Return the PyTorch Dataset

    else:
        raise ValueError(f"Unknown regression dataset name: {dataset_name}. "
                         f"Supported datasets are: {REGRESSION_DATASETS}")
