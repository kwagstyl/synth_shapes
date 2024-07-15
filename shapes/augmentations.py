import numpy as np
from scipy import ndimage

def add_low_frequency_noise(image):
    """
    Add low frequency noise to an image with random parameters.
    
    Parameters:
    image (np.array): Input image as a numpy array.
    
    Returns:
    np.array: Image with added low frequency noise.
    """
    strength = np.random.uniform(0.05, 0.3)
    frequency = np.random.randint(3, 20)
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    noise = np.sin(x / frequency) * np.sin(y / frequency)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = (noise - 0.5) * 2 * strength
    
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def add_high_frequency_noise(image):
    """
    Add high frequency (salt and pepper) noise to an image with random strength.
    
    Parameters:
    image (np.array): Input image as a numpy array.
    
    Returns:
    np.array: Image with added high frequency noise.
    """
    strength = np.random.uniform(0.01, 0.1)
    noise = np.random.choice([-1, 0, 1], size=image.shape, p=[strength/2, 1-strength, strength/2])
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def apply_gamma_correction(image):
    """
    Apply gamma correction to an image with random gamma value.
    
    Parameters:
    image (np.array): Input image as a numpy array.
    
    Returns:
    np.array: Gamma-corrected image.
    """
    gamma = np.random.uniform(0.5, 2.0)
    return np.power(image, gamma)

def gaussian_blur(image):
    """
    Apply Gaussian blur to an image with random sigma.
    
    Parameters:
    image (np.array): Input image as a numpy array.
    
    Returns:
    np.array: Blurred image.
    """
    sigma = np.random.uniform(0.5, 3.0)
    return ndimage.gaussian_filter(image, sigma=sigma)

def sharpen(image):
    """
    Sharpen an image using an unsharp mask with random strength.
    
    Parameters:
    image (np.array): Input image as a numpy array.
    
    Returns:
    np.array: Sharpened image.
    """
    strength = np.random.uniform(0.5, 2.0)
    blurred = gaussian_blur(image)
    sharpened = image + strength * (image - blurred)
    return np.clip(sharpened, 0, 1)

def invert_colors(image):
    """
    Invert the colors of an image.
    """
    return 1 - image

def random_augment(image):
    """
    Apply augmentations to the image, each with a 20% probability.
    
    Parameters:
    image (np.array): Input image as a numpy array.
    
    Returns:
    np.array: Augmented image.
    list: List of applied augmentation names.
    """
    augmentations = [
        (add_low_frequency_noise, "Low Frequency Noise"),
        (add_high_frequency_noise, "High Frequency Noise"),
        (apply_gamma_correction, "Gamma Correction"),
        (gaussian_blur, "Gaussian Blur"),
        (sharpen, "Sharpen"),
        (invert_colors, "Invert Colors")
    ]
    
    applied_augmentations = []
    augmented_image = image.copy()
    
    for aug_func, aug_name in augmentations:
        if np.random.random() < 0.5:  # 20% probability
            augmented_image = aug_func(augmented_image)
            applied_augmentations.append(aug_name)
    
    return augmented_image