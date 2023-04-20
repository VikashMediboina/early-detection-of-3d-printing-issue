import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Define the transforms
dataTransform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to a fixed size
    # transforms.Pad((100, 100), fill=0),   # Pad with zeros to match the desired size
    # transforms.RandomHorizontalFlip(),   # Apply horizontal flip randomly
    # transforms.RandomRotation(10),   # Rotate the image randomly by up to 10 degrees
    transforms.ToTensor(),   # Convert the image to a tensor
    # transforms.Normalize([0.7561, 0.7167, 0.6854], [0.2532, 0.2650, 0.2840])   # Normalize the pixel values

])

class CachedDataset(Dataset):
    def __init__(self, data, save_dir=None):
        self.data = data
        self.cache = {}
        self.save_dir = save_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            x, y = self.data[idx]
            if isinstance(x, torch.Tensor):
                x = transforms.ToPILImage()(x)  # Convert tensor to PIL image
            transformed_x = dataTransform(x)
            if self.save_dir is not None:
                filename = 'image_{}.jpg'.format(idx)
                save_path = os.path.join(self.save_dir, filename)
                transformed_image = transforms.ToPILImage()(transformed_x)
                transformed_image.save(save_path)
            self.cache[idx] = transformed_x, y
            return transformed_x, y

class CachedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, save_dir=None):
        super().__init__(
            dataset=CachedDataset(dataset, save_dir=save_dir),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )
        self.save_dir = save_dir

    def collate_fn(self, batch):
        if isinstance(batch[0], tuple):
            images, labels = zip(*batch)
        else:
            images = batch
            labels = None
        images = torch.stack(images)
        if labels is not None:
            labels = torch.tensor(labels)
        return images, labels



def getData(batch_size, train_split):
    # Set the paths for the train and validation sets
    data_dir = './data'
    save_dir = './augmented_images'

    # Load the train and validation datasets using ImageFolder
    data = datasets.ImageFolder(data_dir, transform=dataTransform)
    # Split the train dataset to get a smaller train set and a validation set
    train_size = int(train_split * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    # Create a directory to save augmented images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Create the data loaders for train and val using CachedDataLoader
    train_loader = CachedDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, save_dir=None)
    val_loader = CachedDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, save_dir=None)
    return train_loader, val_loader
    