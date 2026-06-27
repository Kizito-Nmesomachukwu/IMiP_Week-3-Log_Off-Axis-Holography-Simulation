import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from skimage.transform import iradon
from skimage.restoration import unwrap_phase
from skimage.measure import marching_cubes
import pyvista as pv


def make_3d_phantom(N=128):
    z, y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]

    vol = np.zeros_like(x)

    # Main ellipsoid body
    vol[(x / 0.55) ** 2 + (y / 0.18) ** 2 + (z / 0.12) ** 2 <= 1] = 1.0

    # Inner dense core
    vol[(x / 0.12) ** 2 + (y / 0.07) ** 2 + (z / 0.05) ** 2 <= 1] = 2.0

    # Wing-like planar structures
    wing1 = (
        (np.abs(y - 0.30) <= 0.04)
        & (np.abs(x) <= 0.65)
        & (np.abs(z) <= 0.08)
    )
    vol[wing1] = 0.7

    wing2 = (
        (np.abs(y + 0.30) <= 0.04)
        & (np.abs(x) <= 0.65)
        & (np.abs(z) <= 0.08)
    )
    vol[wing2] = 0.7

    # Texture / fine structure
    texture = 0.15 * np.sin(70 * x + 20 * np.sin(8 * y))
    vol += texture * (vol > 0.15)

    vol = gaussian_filter(vol, sigma=1.2)
    vol -= vol.min()
    vol /= vol.max()
    return vol


def add_phase_artifacts(phase, angle_deg):
    N = phase.shape[0]
    y, x = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]

    ripple = 0.15 * np.sin(45 * x + 25 * y + np.deg2rad(angle_deg))
    background = 0.3 * x + 0.15 * y
    noise = 0.08 * np.random.randn(N, N)

    return phase + ripple + background + noise


def generate_projections(volume, num_angles=60, max_phase_rad=8.0):
    N = volume.shape[0]
    angles = np.linspace(0, 180, num_angles, endpoint=False)

    projections = []
    for angle in angles:
        # Rotate 3D volume around the Y axis (vertical)
        rotated = rotate(volume, angle, axes=(0, 2), reshape=False, order=1, mode="constant")
        # Sum along X axis → line integral = phase projection
        proj = rotated.sum(axis=0)
        # Scale to desired phase range
        phase = max_phase_rad * proj / (proj.max() + 1e-12)

        phase = add_phase_artifacts(phase, angle)
        wrapped = np.angle(np.exp(1j * phase))
        unwrapped = unwrap_phase(wrapped)
        projections.append(unwrapped.astype(np.float32))

    return np.array(projections), angles


def show_projections(proj_stack, angles, n_show=12):
    idxs = np.linspace(0, len(angles) - 1, n_show, dtype=int)
    cols = 4
    rows = int(np.ceil(n_show / cols))
    vmin = np.percentile(proj_stack, 2)
    vmax = np.percentile(proj_stack, 98)

    plt.figure(figsize=(14, 3.5 * rows))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(proj_stack[idx], cmap="viridis", vmin=vmin, vmax=vmax)
        plt.title(f"Angle {angles[idx]:.1f}°")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def prepare_phase_stack(phase_stack):
    phase_stack = np.asarray(phase_stack, dtype=np.float32)
    phase_stack = phase_stack - np.median(phase_stack, axis=(1, 2), keepdims=True)
    phase_stack = phase_stack - phase_stack.min()
    phase_stack = phase_stack / (phase_stack.max() + 1e-12)
    return phase_stack


def reconstruct_volume(proj_stack, angles):
    proj_stack = prepare_phase_stack(proj_stack)
    num_angles, H, W = proj_stack.shape

    if len(angles) != num_angles:
        raise ValueError("len(angles) must equal proj_stack.shape[0]")

    recon = np.zeros((H, W, W), dtype=np.float32)
    for row in range(H):
        sino = proj_stack[:, row, :].T
        rec_slice = iradon(sino, theta=angles, filter_name="ramp", circle=False, output_size=W)
        recon[row, :, :] = rec_slice.astype(np.float32)

    recon -= recon.min()
    recon /= recon.max() + 1e-12
    return recon


def show_slices(volume, title="Reconstructed volume"):
    Z, Y, X = volume.shape
    zmid, ymid, xmid = Z // 2, Y // 2, X // 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(volume[zmid, :, :], cmap="gray")
    plt.title("XY slice (z-mid)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(volume[:, ymid, :], cmap="gray")
    plt.title("XZ slice (y-mid)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(volume[:, :, xmid], cmap="gray")
    plt.title("YZ slice (x-mid)")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def render_3d(volume, threshold=0.25, opacity=0.7, show_edges=False):
    try:
        verts, faces, normals, values = marching_cubes(volume, level=threshold)
    except ValueError:
        print(f"No surface found at threshold {threshold}. Try lowering the threshold.")
        return

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(verts, faces_pv)

    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.add_mesh(
        mesh,
        opacity=opacity,
        smooth_shading=True,
        show_edges=show_edges,
        cmap="viridis",
        scalars=normals[:, 2],
    )
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.show_grid()
    plotter.show()


def plot_ground_truth_slices(volume, title="Ground truth volume"):
    show_slices(volume, title=title)


def plot_comparison(ground_truth, reconstructed, title="Ground truth vs reconstruction"):
    gt_slice = ground_truth[ground_truth.shape[0] // 2, :, :]
    rec_slice = reconstructed[reconstructed.shape[0] // 2, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(gt_slice, cmap="viridis")
    axes[0].set_title("Ground truth (mid slice)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rec_slice, cmap="viridis")
    axes[1].set_title("Reconstructed (mid slice)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    diff = np.abs(gt_slice - rec_slice)
    im2 = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Absolute difference")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
