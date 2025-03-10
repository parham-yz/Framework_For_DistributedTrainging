import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data.dataset import to_map_style_dataset
import torch

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
    


def generate_nlp_data(dataset_name, batch_size=32, max_vocab_size=10000, max_length=512):
    if dataset_name.lower() != "imdb":
        raise ValueError("Currently, only the 'IMDB' dataset is supported.")

    # Load the IMDB dataset
    train_iter, test_iter = IMDB()

    # Tokenization
    tokenizer = get_tokenizer('basic_english')

    # Build vocabulary
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])  # Set default index for unknown words

    # Limit the size of the vocabulary
    if len(vocab) > max_vocab_size:
        vocab = vocab.freq_cutoff(max_vocab_size - 2)  # Adjust for special tokens

    # Numericalize and pad text function
    def process_text(text):
        tokenized_text = tokenizer(text)
        numericalized_text = vocab(tokenized_text)
        padded_text = numericalized_text[:max_length] + [vocab['<pad>']] * (max_length - len(numericalized_text))
        return torch.tensor(padded_text)

    # Process the datasets
    def collate_batch(batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_list.append(1 if label == 'pos' else 0)
            processed_text = process_text(text)
            text_list.append(processed_text)
        return torch.tensor(label_list), torch.stack(text_list)

    # Convert to DataLoader
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_dataloader, test_dataloader
