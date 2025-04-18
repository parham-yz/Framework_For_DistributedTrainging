
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import random

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
]
REGRESSION_DATASETS = ["ones"]
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


def generate_regressiondata(dataset_name, n_samples=1000, d1=10, d2=1):
    """
    Generates a synthetic regression dataset.
    Currently, only the "ones" dataset is supported.
    For each sample:
      - The input is a tensor of ones with shape (d1,)
      - The output is a tensor of ones with shape (d2,)
    
    Parameters:
        dataset_name (str): Name of the regression dataset. Currently, only "ones" is supported.
        n_samples (int): Number of samples in the dataset (default is 1000).
        d1 (int): Dimension of the input features (default is 10).
        d2 (int): Dimension of the output (default is 1).
    
    Returns:
        torch.utils.data.TensorDataset: A TensorDataset where each sample is a tuple (input, target)
        with input of shape (d1,) and target of shape (d2,).
    """
    if dataset_name != "ones":
        raise ValueError("Unknown regression dataset name. Currently, only 'ones' is supported.")
    
    # Create input and output tensors filled with ones
    X = torch.ones(n_samples, d1)
    y = torch.ones(n_samples, d2)
    
    # Create a TensorDataset from the generated tensors
    dataset = torch.utils.data.TensorDataset(X, y)
    return dataset