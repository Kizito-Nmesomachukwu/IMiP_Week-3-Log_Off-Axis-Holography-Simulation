import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from PIL import Image

def load_hologram(path, size=512):
    img = Image.open(path).convert('L').resize((size, size))
    return np.array(img, dtype=np.float64)


def load_phase_image(path, size=512, phase_range=2.0):
    img = Image.open(path).convert('L').resize((size, size))
    phase = np.array(img, dtype=np.float64)
    phase = phase / phase.max() * phase_range
    return phase


def generate_test_phase(N=512, num_features=8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, Y = make_grid(N)
    phase = np.zeros((N, N))
    for _ in range(num_features):
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        sigma = np.random.uniform(0.05, 0.3)
        amp = np.random.uniform(0.5, 3.0)
        phase += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    return phase


def make_grid(N=512):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    return np.meshgrid(x, y)


def generate_hologram(true_phase, kx=20, ky=20, X=None, Y=None):
    N = true_phase.shape[0]
    if X is None or Y is None:
        X, Y = make_grid(N)
    amplitude = np.ones_like(true_phase)
    reference = np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    object_wave = amplitude * np.exp(1j * true_phase)
    hologram = np.abs(reference + object_wave) ** 2
    return hologram, reference, object_wave


def compute_fft(hologram):
    return np.fft.fftshift(np.fft.fft2(hologram))


def isolate_sideband(H, radius=30, dc_block=15):
    N = H.shape[0]
    H_mag = np.abs(H).copy()
    cy_centre = cx_centre = N // 2
    H_mag[cy_centre - dc_block: cy_centre + dc_block,
          cx_centre - dc_block: cx_centre + dc_block] = 0
    cy, cx = np.unravel_index(np.argmax(H_mag), H_mag.shape)
    print(f"Sideband at pixel: row={cy}, col={cx}  "
          f"(DC centre at {cy_centre},{cx_centre}; "
          f"offset = {cy - cy_centre}, {cx - cx_centre})")
    rows_idx = np.arange(N)[:, None]
    cols_idx = np.arange(N)[None, :]
    mask = (rows_idx - cy) ** 2 + (cols_idx - cx) ** 2 <= radius ** 2
    H_filtered = H * mask
    return H_filtered, H_mag, cy, cx


def reconstruct_complex_field(H_filtered, cy, cx):
    N = H_filtered.shape[0]
    H_centred = np.roll(np.roll(H_filtered, N // 2 - cy, axis=0), N // 2 - cx, axis=1)
    return np.fft.ifft2(np.fft.ifftshift(H_centred))


def remove_background_tilt(phase, order=1):
    X, Y = np.meshgrid(np.arange(phase.shape[1]), np.arange(phase.shape[0]))
    mask = np.isfinite(phase)
    A = np.c_[X[mask], Y[mask], np.ones(mask.sum())]
    if order >= 2:
        A = np.c_[X[mask]**2, Y[mask]**2, X[mask]*Y[mask], A]
    coeffs, _, _, _ = np.linalg.lstsq(A, phase[mask], rcond=None)
    bg = np.zeros_like(phase)
    if order == 1:
        bg = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    elif order >= 2:
        bg = (coeffs[0] * X**2 + coeffs[1] * Y**2 + coeffs[2] * X * Y +
              coeffs[3] * X + coeffs[4] * Y + coeffs[5])
    return phase - bg


def extract_phase(complex_field, remove_tilt=True):
    wrapped_phase = np.angle(complex_field)
    unwrapped_phase = unwrap_phase(wrapped_phase)
    if remove_tilt:
        unwrapped_phase = remove_background_tilt(unwrapped_phase, order=1)
    return wrapped_phase, unwrapped_phase


def reconstruct_phase(hologram, radius=30, dc_block=15):
    H = compute_fft(hologram)
    H_filtered, H_mag, cy, cx = isolate_sideband(H, radius, dc_block)
    complex_field = reconstruct_complex_field(H_filtered, cy, cx)
    wrapped_phase, unwrapped_phase = extract_phase(complex_field)
    return unwrapped_phase, wrapped_phase, H, H_filtered, complex_field


def compute_error(true_phase, reconstructed_phase):
    phase_recon = reconstructed_phase - np.mean(reconstructed_phase)
    phase_truth = true_phase - np.mean(true_phase)
    error_map = phase_recon - phase_truth
    rmse = np.sqrt(np.mean(error_map ** 2))
    print(f"Reconstruction RMSE: {rmse:.4f} rad")
    return phase_recon, phase_truth, error_map, rmse


def plot_hologram(hologram):
    plt.figure(figsize=(5, 4))
    plt.imshow(hologram, cmap='gray')
    plt.title('Hologram (Interference Pattern)')
    plt.axis('off')
    plt.show()


def plot_spectrum(H):
    plt.figure(figsize=(5, 4))
    plt.imshow(np.log(1 + np.abs(H)), cmap='gray')
    plt.title('FFT Spectrum (DC + Sidebands)')
    plt.axis('off')
    plt.show()


def plot_sideband_selection(H_mag, H_filtered):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.log(1 + np.abs(H_mag)), cmap='gray')
    axes[0].set_title('DC Blocked; Sideband Highlighted')
    axes[0].axis('off')
    axes[1].imshow(np.log(1 + np.abs(H_filtered)), cmap='gray')
    axes[1].set_title('Isolated +1 Sideband')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def plot_reconstructed_amplitude(complex_field):
    plt.figure(figsize=(5, 4))
    plt.imshow(np.abs(complex_field), cmap='gray')
    plt.title('Reconstructed Amplitude')
    plt.axis('off')
    plt.show()


def plot_phase_comparison(wrapped_phase, unwrapped_phase):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(wrapped_phase, cmap='hsv')
    axes[0].set_title('Wrapped Phase (-π to +π)')
    axes[0].axis('off')
    im = axes[1].imshow(unwrapped_phase, cmap='viridis')
    axes[1].set_title('Unwrapped Phase (rad)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_input_phase(phase):
    plt.figure(figsize=(5, 4))
    plt.imshow(phase, cmap='viridis')
    plt.title('Input Phase Map (rad)')
    plt.colorbar()
    plt.axis('off')
    plt.show()


def plot_error_map(error_map, rmse):
    plt.figure(figsize=(5, 4))
    plt.imshow(error_map, cmap='RdBu', vmin=-0.5, vmax=0.5)
    plt.title(f'Reconstruction Error (RMSE = {rmse:.4f} rad)')
    plt.colorbar()
    plt.axis('off')
    plt.show()


def plot_results(hologram, H, unwrapped_phase, save_path="pipeline_visualisation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Off-Axis Holographic Phase Retrieval Pipeline", fontsize=14)

    axes[0].imshow(hologram, cmap='gray')
    axes[0].set_title('Hologram')

    axes[1].imshow(np.log(1 + np.abs(H)), cmap='gray')
    axes[1].set_title('FFT Spectrum')

    im = axes[2].imshow(unwrapped_phase, cmap='viridis')
    axes[2].set_title('Reconstructed Phase (rad)')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Pipeline visualisation saved as {save_path}")


def run_pipeline(image_path="img/pattern_01.png", N=None,
                 radius=30, dc_block=15,
                 save_path="pipeline_visualisation.png"):
    hologram = load_hologram(image_path, size=N)
    unwrapped_phase, wrapped_phase, H, _, _ = reconstruct_phase(
        hologram, radius=radius, dc_block=dc_block
    )
    plot_results(hologram, H, unwrapped_phase, save_path=save_path)
    return {
        "hologram": hologram,
        "unwrapped_phase": unwrapped_phase,
        "wrapped_phase": wrapped_phase,
    }


if __name__ == "__main__":
    results = run_pipeline()
