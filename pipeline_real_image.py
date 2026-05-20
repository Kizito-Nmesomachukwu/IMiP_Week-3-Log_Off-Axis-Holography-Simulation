# Off-Axis Holographic Phase Retrieval Pipeline: Real Image Ingestion and FFT Processing Test


import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from skimage import io, color

# numpy handles all the heavy math and matrix grids.
# matplotlib is used to draw the pictures at the end.
# unwrap_phase from skimage is a special tool to fix phase maps that jump abruptly.
# io and color from skimage allow us to load real image files and convert them to grayscale.

# =============================================================================
# Title: Off-Axis Holographic Phase Retrieval Pipeline
#        Modified to accept real experimental images as input
# =============================================================================
# Description:
# This is a modified version of the original synthetic simulation pipeline.
# Instead of generating a fake Gaussian blob as the sample, this version
# loads a real image file from disk and runs it through the same FFT-based
# phase retrieval steps.
#
# I built this version for two reasons. First, to test whether the pipeline
# code itself runs correctly on real pixel data before we have a clean
# experimental hologram to work with. Second, to demonstrate what happens
# when a real photograph with no off-axis carrier frequency is fed into the
# algorithm. The result is instructive: without a carrier, there are no
# sidebands in the FFT spectrum, so the reconstruction produces nothing
# meaningful. This is not a bug. It is the expected physical behaviour and
# it directly shows why the off-axis mirror angle in the lab matters so much.
#
# When we eventually have a properly aligned hologram from the Mach-Zehnder
# setup, this script can be used to process it by simply changing the
# image_name variable at the top to point to that file.

# =============================================================================
# Background
# =============================================================================
# In off-axis digital holography, the reference beam hits the camera at a
# small angle relative to the object beam. This angle acts as a carrier
# frequency that pushes the phase information into a sideband in the Fourier
# domain, away from the bright DC term in the centre. The pipeline isolates
# that sideband, shifts it to the centre, and runs an inverse FFT to recover
# the complex field. The phase of that field contains the optical path
# difference introduced by the sample, which is proportional to its
# refractive index.
#
# A natural photograph has none of this. Its spatial frequency content sits
# entirely in the DC region with no imposed carrier separation. Running it
# through this pipeline therefore produces a flat, near-zero phase map,
# which is exactly what we observed and is documented in Week 4 of the log.

# =============================================================================
# Known Limitations
# =============================================================================
# Since we do not have a ground truth phase map for a real image, the RMSE
# reported at the end is computed against a zero baseline. This number is
# not a meaningful accuracy metric for real images. It is kept in the output
# only so the code structure stays consistent with the synthetic version.
#
# The DC block radius of 15 pixels and mask radius of 30 pixels were tuned
# for a 512x512 synthetic hologram. For a real experimental hologram at a
# different resolution or fringe density, these values may need adjustment.
# Check the FFT spectrum visually first before committing to fixed values.

# =============================================================================
# Dependencies
# =============================================================================
# numpy, matplotlib, scikit-image
# All three come pre-installed in Google Colab. No pip install needed.

# =============================================================================
# Parameters you can change
# =============================================================================
# image_name   : file name of the image you want to process. Must be in the
#                same folder as this script, or provide the full path.
# dc_block     : half-width in pixels of the box zeroed around the DC term.
# radius       : radius in pixels of the circular sideband filter.

# =============================================================================
# Step 1: Load real image
# =============================================================================
# Change this to the name of your hologram file.
# Supported formats: jpg, png, tif, bmp
image_name = 'kizito.jpg'

# Load the image and convert to grayscale if it is RGB or RGBA.
# The pipeline needs a 2D array, not a 3-channel colour image.
raw_image = io.imread(image_name)
if raw_image.ndim == 3:
    hologram = color.rgb2gray(raw_image)
elif raw_image.ndim == 4:
    hologram = color.rgb2gray(raw_image[:, :, :3])
else:
    hologram = raw_image.astype(float)

# Crop to a square using the shorter side.
# This keeps the FFT grid square, which is required for the N//2 centring to work.
N = min(hologram.shape)
hologram = hologram[:N, :N]

# We have no ground truth for a real image so we set it to zero.
# This is only here so the plotting code at the end does not crash.
# The error map in panel 6 will look identical to the phase reconstruction,
# which is expected and noted in the figure title.
true_phase = np.zeros((N, N))

# =============================================================================
# Step 2: 2D Fast Fourier Transform
# =============================================================================
# fft2 converts our image from pixel space into spatial frequency space.
# fftshift moves the DC term to the centre of the image so it is easier
# to mask out in the next step.
H = np.fft.fftshift(np.fft.fft2(hologram))

# =============================================================================
# Step 3: Locate and isolate the sideband
# =============================================================================
# Take the magnitude so we can search for peak brightness.
H_mag = np.abs(H).copy()

# Zero out the DC region at the centre so the search ignores it.
# Without this, argmax just returns the centre pixel every time.
dc_block = 15
cy_centre, cx_centre = N // 2, N // 2
H_mag[cy_centre - dc_block : cy_centre + dc_block,
      cx_centre - dc_block : cx_centre + dc_block] = 0

# Search only the top-left quadrant of the spectrum.
# The hologram is a real-valued image so its spectrum is Hermitian.
# The top-left quadrant holds the sideband that carries +object_phase.
# The bottom-right holds the conjugate, which gives -object_phase.
# Restricting the search here means we always get the correct sign.
# For a real photograph with no carrier this search will land on a noise
# peak, which is expected. The reconstruction will be flat as a result.
H_search = H_mag.copy()
H_search[N//2:, :] = 0
H_search[:, N//2:] = 0
cy, cx = np.unravel_index(np.argmax(H_search), H_mag.shape)
print(f"Sideband located at pixel: row={cy}, col={cx}  "
      f"(DC centre is at {cy_centre}, {cx_centre}; "
      f"offset = {cy - cy_centre}, {cx - cx_centre})")

# Apply a circular mask around the found peak.
# For a real hologram this isolates the sideband lobe.
# For a plain photograph it isolates a noise peak, which is fine.
# The output in that case is just a flat phase map, as documented in the log.
radius   = 30
rows_idx = np.arange(N)[:, None]
cols_idx = np.arange(N)[None, :]
mask     = (rows_idx - cy)**2 + (cols_idx - cx)**2 <= radius**2
H_filtered = H * mask

# =============================================================================
# Step 4: Centre sideband at DC, then inverse FFT
# =============================================================================
# Shift the sideband to the centre of the frequency grid.
# This removes the carrier tilt from the reconstruction.
# Without this step the phase output contains a strong diagonal gradient
# from the off-axis angle, and the object phase is buried under it.
H_centred     = np.roll(np.roll(H_filtered, N // 2 - cy, axis=0),
                                             N // 2 - cx, axis=1)
complex_field = np.fft.ifft2(np.fft.ifftshift(H_centred))

# =============================================================================
# Step 5: Extract and unwrap phase
# =============================================================================
# np.angle gives the phase in the range [-pi, +pi].
# This produces sharp concentric rings at every 2-pi boundary.
wrapped_phase = np.angle(complex_field)

# unwrap_phase removes those jumps to give a continuous smooth surface.
unwrapped_phase = unwrap_phase(wrapped_phase)

# =============================================================================
# Step 6: Reconstruction accuracy
# =============================================================================
# Mean-subtract the reconstruction so it starts at a zero baseline.
# For real images the ground truth is zero so the error map and the
# phase reconstruction will look the same. This is expected behaviour.
phase_recon = unwrapped_phase - np.mean(unwrapped_phase)
phase_truth = true_phase - np.mean(true_phase)
error_map   = phase_recon - phase_truth
rmse        = np.sqrt(np.mean(error_map**2))
print(f"Reconstruction RMSE: {rmse:.4f} rad")
print("Note: for real images the ground truth is zero, so RMSE here is not")
print("a meaningful accuracy number. It just shows the phase variation present.")

# =============================================================================
# Step 7: Visualise
# =============================================================================
# Centre-crop 100x100 pixels for the hologram zoom panel.
zoom_start = (N // 2) - 50
zoom_end   = (N // 2) + 50
crop       = slice(zoom_start, zoom_end)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Off-Axis Phase Retrieval Pipeline: Real Image Input Test", fontsize=14)

axes[0, 0].imshow(hologram[crop, crop], cmap='gray')
axes[0, 0].set_title('1. Input Image (Zoomed)')

# Log scale so faint sidebands (if any) are visible alongside the bright DC.
axes[0, 1].imshow(np.log(1 + np.abs(H)), cmap='gray')
axes[0, 1].set_title('2. FFT Spectrum')

axes[0, 2].imshow(true_phase, cmap='viridis')
axes[0, 2].set_title('3. Ground Truth (not available for real images)')

# HSV colormap makes the 2-pi wrapping jumps show up as sudden colour shifts.
axes[1, 0].imshow(wrapped_phase, cmap='hsv')
axes[1, 0].set_title('4. Wrapped Phase (-pi to +pi)')

axes[1, 1].imshow(unwrapped_phase, cmap='viridis')
axes[1, 1].set_title('5. Unwrapped Phase (rad)')

# For real images this panel is the same as panel 5 since ground truth is zero.
im = axes[1, 2].imshow(error_map, cmap='RdBu')
axes[1, 2].set_title('6. Processed Phase Map\n(same as panel 5 for real images)')
fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('pipeline_visualisation_real_image.png', dpi=150, bbox_inches='tight')
plt.show()
print("Pipeline complete. Image saved as pipeline_visualisation_real_image.png")
