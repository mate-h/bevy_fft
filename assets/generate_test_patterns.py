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

def create_colorful_pattern(size=(256, 256), frequency=(5, 5)):
    """Create a colorful pattern with different frequencies in each channel."""
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Create different patterns for each RGB channel
    r_pattern = np.sin(2 * np.pi * frequency[0] * xx) * np.cos(2 * np.pi * frequency[1] * yy)
    g_pattern = np.sin(2 * np.pi * (frequency[0] + 3) * xx) * np.sin(2 * np.pi * frequency[1] * yy)
    b_pattern = np.cos(2 * np.pi * frequency[0] * xx) * np.sin(2 * np.pi * (frequency[1] + 3) * yy)
    
    # Scale to 0-255
    r_channel = ((r_pattern + 1) / 2 * 255).astype(np.uint8)
    g_channel = ((g_pattern + 1) / 2 * 255).astype(np.uint8)
    b_channel = ((b_pattern + 1) / 2 * 255).astype(np.uint8)
    
    # Combine into RGB
    rgb_pattern = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    return rgb_pattern

def create_rainbow_spiral(size=(256, 256), revolutions=3):
    """Create a colorful rainbow spiral pattern."""
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    
    # Create spiral pattern (phase varies with radius and angle)
    phase = theta + revolutions * 2 * np.pi * r
    
    # Create RGB channels with phase offsets for rainbow effect
    r_channel = ((np.sin(phase) + 1) / 2 * 255).astype(np.uint8)
    g_channel = ((np.sin(phase + 2*np.pi/3) + 1) / 2 * 255).astype(np.uint8)
    b_channel = ((np.sin(phase + 4*np.pi/3) + 1) / 2 * 255).astype(np.uint8)
    
    # Apply radial fade to create a smooth edge
    fade = np.clip(1.0 - r, 0, 1)**0.5
    r_channel = (r_channel * fade).astype(np.uint8)
    g_channel = (g_channel * fade).astype(np.uint8)
    b_channel = (b_channel * fade).astype(np.uint8)
    
    # Combine into RGB
    rgb_pattern = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    return rgb_pattern

def create_color_mandelbrot(size=(256, 256), max_iter=100):
    """Create a colorful Mandelbrot set visualization."""
    x = np.linspace(-2, 1, size[0])
    y = np.linspace(-1.5, 1.5, size[1])
    xx, yy = np.meshgrid(x, y)
    c = xx + 1j * yy
    
    z = np.zeros_like(c)
    output = np.zeros(size, dtype=np.int32)
    
    # Compute Mandelbrot set
    for i in range(max_iter):
        mask = np.abs(z) < 2
        z[mask] = z[mask]**2 + c[mask]
        output[mask & (np.abs(z) >= 2)] = i
    
    # Create smooth coloring
    output = output.astype(np.float32) / max_iter
    
    # Map to RGB using a colorful gradient
    r_channel = (np.sin(output * 2 * np.pi) * 127 + 128).astype(np.uint8)
    g_channel = (np.sin(output * 4 * np.pi) * 127 + 128).astype(np.uint8)
    b_channel = (np.sin(output * 6 * np.pi) * 127 + 128).astype(np.uint8)
    
    # Combine into RGB
    rgb_pattern = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    return rgb_pattern

def visualize_fft(image_data):
    """Compute and visualize the FFT of an image with separate horizontal and vertical passes."""
    # Convert to grayscale if RGB
    if len(image_data.shape) == 3:
        gray_data = np.mean(image_data, axis=2)
    else:
        gray_data = image_data
    
    # Horizontal pass only (FFT along rows)
    f_horizontal = np.zeros_like(gray_data, dtype=np.complex128)
    for i in range(gray_data.shape[0]):
        f_horizontal[i, :] = np.fft.fft(gray_data[i, :])
    f_h_shift = np.fft.fftshift(f_horizontal)
    
    # Complete 2D FFT (both passes)
    f_transform = np.fft.fft2(gray_data)
    f_shift = np.fft.fftshift(f_transform)
    
    # Compute magnitude spectrums
    magnitude_h = 20 * np.log(np.abs(f_h_shift) + 1)
    magnitude_full = 20 * np.log(np.abs(f_shift) + 1)
    
    # Get real and imaginary parts of full FFT
    real_part = np.real(f_shift)
    imag_part = np.imag(f_shift)
    
    # Normalize for visualization
    def normalize(data):
        data_range = np.max(data) - np.min(data)
        if data_range > 0:
            return (data - np.min(data)) / data_range * 255
        return np.zeros_like(data)
    
    real_normalized = normalize(real_part)
    imag_normalized = normalize(imag_part)
    
    return magnitude_h, magnitude_full, real_normalized, imag_normalized

def compare_roots():
    # Our implementation
    def calculate_root(k, base):
        theta = -2.0 * np.pi * k / base
        return np.cos(theta) + 1j * np.sin(theta)
    
    # NumPy's implementation
    def numpy_root(k, N):
        return np.exp(-2j * np.pi * k / N)
    
    # Compare for a few values
    base = 8
    for k in range(base//2):
        your_root = calculate_root(k, base)
        np_root = numpy_root(k, base)
        print(f"k={k}: Ours={your_root:.4f}, NumPy={np_root:.4f}")

def main():
    # Set image size
    size = (256, 256)
    
    # Create and save test patterns
    patterns = {
        "sine": create_sine_pattern(size, frequency=(8, 8)),
        "impulse": create_impulse_pattern(size),
        "checkerboard": create_checkerboard_pattern(size, check_size=32),
        "circles": create_concentric_circles(size, num_circles=8),
        "stripes": create_diagonal_stripes(size, stripe_width=16),
        "colorful": create_colorful_pattern(size, frequency=(5, 5)),
        "rainbow_spiral": create_rainbow_spiral(size, revolutions=3),
        "mandelbrot": create_color_mandelbrot(size, max_iter=100)
    }
    
    # Save each pattern and its FFT components
    for name, pattern in patterns.items():
        # Save original pattern
        save_image(pattern, f"{name}_pattern.png", normalize=False)
        
        # Compute and visualize FFT
        magnitude_h, magnitude_full, real, imag = visualize_fft(pattern)
        
        # Create a combined visualization
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(pattern)
        axes[0].set_title("Original Pattern")
        axes[1].imshow(magnitude_h, cmap='viridis')
        axes[1].set_title("After Horizontal Pass")
        axes[2].imshow(magnitude_full, cmap='viridis')
        axes[2].set_title("Full FFT Magnitude")
        axes[3].imshow(real, cmap='viridis')
        axes[3].set_title("FFT Real Part")
        axes[4].imshow(imag, cmap='viridis')
        axes[4].set_title("FFT Imaginary Part")
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f"{name}_fft_visualization.png")
        plt.close()

    compare_roots()

if __name__ == "__main__":
    main() 