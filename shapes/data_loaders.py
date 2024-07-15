import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


# Import the shape generation functions
from shape_generator import (
    draw_random_filled_circle,
    draw_random_filled_triangle,
    draw_random_filled_square,
    draw_random_filled_rectangle,
    draw_random_filled_ellipse
)
from augmentations import random_augment

class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(128, 128), mode='on-the-fly',
                 augment=True):
        self.num_samples = num_samples
        self.image_size = image_size
        self.shape_functions = [
            draw_random_filled_circle,
            draw_random_filled_triangle,
            draw_random_filled_square,
            draw_random_filled_rectangle,
            draw_random_filled_ellipse
        ]
        self.mode = mode
        self.data = []
        self.augment = augment

        if self.mode == 'fixed':
            self.precompute_dataset()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'fixed':
            item = self.data[idx]
        else:
            item = self.generate_sample()
        if self.augment:
            augmented_image = random_augment(item[0])
# Convert to tensor and add channel dimension
            image_tensor = torch.tensor(augmented_image, dtype=torch.float).unsqueeze(0)
        return image_tensor, item[1],item[2]

    def generate_sample(self):
        # Create a blank image
        image = np.zeros(self.image_size)
        
        # Randomly select a shape function
        shape_function = random.choice(self.shape_functions)
        
        # Generate the shape
        image = shape_function(image)
        
        # get the coordinates of the bounding box
        bbox = self.find_bounding_box(image)
        
        # Create label (index of the shape function)
        label = self.shape_functions.index(shape_function)
        
        return image, label, bbox
    
    def find_bounding_box(self,binary_image):
        """
        Find the bounding box of the object in a binary image.

        Parameters:
        binary_image (np.ndarray): A binary image where the object is represented by non-zero pixels.

        Returns:
        (x, y, dx, dy): Coordinates of the top-left corner of the bounding box and its width and height.
        """
        assert len(binary_image.shape) == 2, "Input image must be a 2D binary image"
        
        # Find non-zero points (object points)
        non_zero_points = np.argwhere(binary_image > 0)

        if non_zero_points.size == 0:
            return None, None, None, None

        # Get the coordinates of the bounding box
        y_min, x_min = non_zero_points.min(axis=0)
        y_max, x_max = non_zero_points.max(axis=0)

        # Calculate the width and height of the bounding box
        dx = x_max - x_min + 1
        dy = y_max - y_min + 1

        return np.array([x_min, y_min, dx, dy],dtype=np.float32)/np.array(self.image_size[0],dtype=np.float32)

    def precompute_dataset(self):
        for _ in range(self.num_samples):
            self.data.append(self.generate_sample())


def get_shape_loaders(num_train=1000, num_test=200,
                       batch_size=32, image_size=(64, 64),
                       mode='on-the-fly'
                      ):
    # Create datasets
    train_dataset = ShapeDataset(num_samples=num_train, 
                                 image_size=image_size,
                                 mode=mode)
    test_dataset = ShapeDataset(num_samples=num_test,
                                 image_size=image_size,
                                 mode=mode)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader