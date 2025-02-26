import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_image(data, filename, normalize=True):
    """Save a numpy array as an image file."""
    # Normalize to 0-255 range if requested
    if normalize:
        data = data - np.min(data)
        if np.max(data) > 0:
            data = data / np.max(data) * 255
    
    # Convert to uint8
    img_data = data.astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(img_data)
    img.save(filename)
    print(f"Saved {filename}")

def create_sine_pattern(size=(256, 256), frequency=(10, 10)):
    """Create a sine wave pattern."""
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Create sine pattern
    pattern = np.sin(2 * np.pi * frequency[0] * xx) * np.sin(2 * np.pi * frequency[1] * yy)
    
    # Scale to 0-255
    pattern = ((pattern + 1) / 2 * 255).astype(np.uint8)
    
    # Convert to RGB
    rgb_pattern = np.stack([pattern, pattern, pattern], axis=2)
    
    return rgb_pattern

def create_impulse_pattern(size=(256, 256), impulse_pos=(128, 128)):
    """Create an impulse (single point) pattern."""
    pattern = np.zeros(size, dtype=np.uint8)
    pattern[impulse_pos] = 255
    
    # Convert to RGB
    rgb_pattern = np.stack([pattern, pattern, pattern], axis=2)
    
    return rgb_pattern

def create_checkerboard_pattern(size=(256, 256), check_size=32):
    """Create a checkerboard pattern."""
    pattern = np.zeros(size, dtype=np.uint8)
    
    for i in range(0, size[0], check_size):
        for j in range(0, size[1], check_size):
            if ((i // check_size) + (j // check_size)) % 2 == 0:
                pattern[i:i+check_size, j:j+check_size] = 255
    
    # Convert to RGB
    rgb_pattern = np.stack([pattern, pattern, pattern], axis=2)
    
    return rgb_pattern

def create_concentric_circles(size=(256, 256), num_circles=10):
    """Create concentric circles pattern."""
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Calculate distance from center
    r = np.sqrt(xx**2 + yy**2)
    
    # Create concentric circles
    pattern = np.cos(2 * np.pi * num_circles * r)
    
    # Scale to 0-255
    pattern = ((pattern + 1) / 2 * 255).astype(np.uint8)
    
    # Convert to RGB
    rgb_pattern = np.stack([pattern, pattern, pattern], axis=2)
    
    return rgb_pattern

def create_diagonal_stripes(size=(256, 256), stripe_width=16):
    """Create diagonal stripes pattern."""
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Create diagonal stripes
    pattern = np.sin(2 * np.pi * (xx + yy) * size[0] / (stripe_width * 2 * np.pi))
    
    # Scale to 0-255
    pattern = ((pattern + 1) / 2 * 255).astype(np.uint8)
    
    # Convert to RGB
    rgb_pattern = np.stack([pattern, pattern, pattern], axis=2)
    
    return rgb_pattern

def visualize_fft(image_data):
    """Compute and visualize the FFT of an image."""
    # Convert to grayscale if RGB
    if len(image_data.shape) == 3:
        gray_data = np.mean(image_data, axis=2)
    else:
        gray_data = image_data
    
    # Compute FFT
    f_transform = np.fft.fft2(gray_data)
    f_shift = np.fft.fftshift(f_transform)
    
    # Compute magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Get real and imaginary parts
    real_part = np.real(f_shift)
    imag_part = np.imag(f_shift)
    
    # Normalize for visualization
    real_normalized = (real_part - np.min(real_part)) / (np.max(real_part) - np.min(real_part)) * 255
    imag_normalized = (imag_part - np.min(imag_part)) / (np.max(imag_part) - np.min(imag_part)) * 255
    
    return magnitude_spectrum, real_normalized, imag_normalized

def main():
    # Set image size
    size = (256, 256)
    
    # Create and save test patterns
    patterns = {
        "sine": create_sine_pattern(size, frequency=(8, 8)),
        "impulse": create_impulse_pattern(size),
        "checkerboard": create_checkerboard_pattern(size, check_size=32),
        "circles": create_concentric_circles(size, num_circles=8),
        "stripes": create_diagonal_stripes(size, stripe_width=16)
    }
    
    # Save each pattern and its FFT components
    for name, pattern in patterns.items():
        # Save original pattern
        save_image(pattern, f"{name}_pattern.png", normalize=False)
        
        # Compute and visualize FFT
        magnitude, real, imag = visualize_fft(pattern)
        
        # Save FFT components
        save_image(magnitude, f"{name}_fft_magnitude.png")
        save_image(real, f"{name}_fft_real.png")
        save_image(imag, f"{name}_fft_imaginary.png")
        
        # Create a combined visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(pattern)
        axes[0].set_title("Original Pattern")
        axes[1].imshow(magnitude, cmap='viridis')
        axes[1].set_title("FFT Magnitude")
        axes[2].imshow(real, cmap='viridis')
        axes[2].set_title("FFT Real Part")
        axes[3].imshow(imag, cmap='viridis')
        axes[3].set_title("FFT Imaginary Part")
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{name}_fft_visualization.png")
        plt.close()

if __name__ == "__main__":
    main() 