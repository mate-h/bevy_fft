import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

out_dir = "assets/test_patterns"
os.makedirs(out_dir, exist_ok=True)

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
    img.save(f"{out_dir}/{filename}")
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
    
    return magnitude_h, magnitude_full, real_normalized, imag_normalized, f_shift

def test_ifft(image_data):
    """Test the round-trip FFT → IFFT process and visualize the results."""
    # Convert to grayscale if RGB
    if len(image_data.shape) == 3:
        gray_data = np.mean(image_data, axis=2)
    else:
        gray_data = image_data
    
    # Compute FFT
    f_transform = np.fft.fft2(gray_data)
    f_shift = np.fft.fftshift(f_transform)
    
    # Apply IFFT to get back the original image
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Calculate error between original and reconstructed image
    error = np.abs(gray_data - img_back)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    # Create test cases with modified frequency domain
    test_cases = {}
    
    # 1. Low-pass filter (remove high frequencies)
    low_pass = f_shift.copy()
    center_y, center_x = low_pass.shape[0] // 2, low_pass.shape[1] // 2
    radius = min(center_y, center_x) // 4
    y, x = np.ogrid[:low_pass.shape[0], :low_pass.shape[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
    low_pass[mask] = 0
    
    low_pass_ishift = np.fft.ifftshift(low_pass)
    low_pass_result = np.fft.ifft2(low_pass_ishift)
    low_pass_result = np.abs(low_pass_result)
    test_cases["low_pass"] = low_pass_result
    
    # 2. High-pass filter (remove low frequencies)
    high_pass = f_shift.copy()
    radius = min(center_y, center_x) // 4
    y, x = np.ogrid[:high_pass.shape[0], :high_pass.shape[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    high_pass[mask] = 0
    
    high_pass_ishift = np.fft.ifftshift(high_pass)
    high_pass_result = np.fft.ifft2(high_pass_ishift)
    high_pass_result = np.abs(high_pass_result)
    test_cases["high_pass"] = high_pass_result
    
    # 3. Band-pass filter
    band_pass = f_shift.copy()
    inner_radius = min(center_y, center_x) // 8
    outer_radius = min(center_y, center_x) // 2
    y, x = np.ogrid[:band_pass.shape[0], :band_pass.shape[1]]
    mask = ((x - center_x)**2 + (y - center_y)**2 <= inner_radius**2) | ((x - center_x)**2 + (y - center_y)**2 > outer_radius**2)
    band_pass[mask] = 0
    
    band_pass_ishift = np.fft.ifftshift(band_pass)
    band_pass_result = np.fft.ifft2(band_pass_ishift)
    band_pass_result = np.abs(band_pass_result)
    test_cases["band_pass"] = band_pass_result
    
    # 4. Phase shift (rotate the image)
    phase_shift = f_shift.copy()
    phase = np.angle(phase_shift)
    magnitude = np.abs(phase_shift)
    phase_rotated = phase + np.pi/4  # 45-degree rotation
    phase_shift_result = magnitude * np.exp(1j * phase_rotated)
    
    phase_shift_ishift = np.fft.ifftshift(phase_shift_result)
    phase_shift_result = np.fft.ifft2(phase_shift_ishift)
    phase_shift_result = np.abs(phase_shift_result)
    test_cases["phase_shift"] = phase_shift_result
    
    # 5. Edge enhancement (high-frequency boost)
    edge_enhance = f_shift.copy()
    y, x = np.ogrid[:edge_enhance.shape[0], :edge_enhance.shape[1]]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    boost_factor = 1 + 2 * (dist_from_center / max_dist)
    edge_enhance = edge_enhance * boost_factor
    
    edge_enhance_ishift = np.fft.ifftshift(edge_enhance)
    edge_enhance_result = np.fft.ifft2(edge_enhance_ishift)
    edge_enhance_result = np.abs(edge_enhance_result)
    test_cases["edge_enhance"] = edge_enhance_result
    
    return img_back, error, max_error, mean_error, test_cases

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
        magnitude_h, magnitude_full, real, imag, f_shift = visualize_fft(pattern)
        
        # Create a combined visualization for FFT
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
        plt.savefig(f"{out_dir}/{name}_fft_visualization.png")
        plt.close()
        
        # Test IFFT and create visualization
        reconstructed, error, max_error, mean_error, test_cases = test_ifft(pattern)
        
        # Create a combined visualization for IFFT
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original and reconstructed
        axes[0, 0].imshow(pattern)
        axes[0, 0].set_title("Original Pattern")
        axes[0, 1].imshow(reconstructed.astype(np.uint8))
        axes[0, 1].set_title(f"Reconstructed (IFFT)\nMax Error: {max_error:.4f}")
        axes[0, 2].imshow(error, cmap='hot')
        axes[0, 2].set_title(f"Error Map\nMean Error: {mean_error:.4f}")
        
        # FFT magnitude for reference
        axes[0, 3].imshow(magnitude_full, cmap='viridis')
        axes[0, 3].set_title("FFT Magnitude")
        
        # Test cases
        axes[1, 0].imshow(test_cases["low_pass"].astype(np.uint8))
        axes[1, 0].set_title("Low-Pass Filter")
        axes[1, 1].imshow(test_cases["high_pass"].astype(np.uint8))
        axes[1, 1].set_title("High-Pass Filter")
        axes[1, 2].imshow(test_cases["band_pass"].astype(np.uint8))
        axes[1, 2].set_title("Band-Pass Filter")
        axes[1, 3].imshow(test_cases["edge_enhance"].astype(np.uint8))
        axes[1, 3].set_title("Edge Enhancement")
        
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f"{out_dir}/{name}_ifft_visualization.png")
        plt.close()
        
        # Save individual test case images
        for case_name, case_img in test_cases.items():
            save_image(case_img, f"{name}_{case_name}.png")

    compare_roots()

if __name__ == "__main__":
    main() 