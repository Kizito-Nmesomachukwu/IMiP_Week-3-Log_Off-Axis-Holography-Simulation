import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

# numpy handles all the heavy math and matrix grids.
# matplotlib is used to draw the pictures at the end.
# unwrap_phase from skimage is a special tool to fix phase maps that jump abruptly.

# =============================================================================
# Step 1: Generate a synthetic off-axis hologram
# =============================================================================
# Set up a square grid of 512 by 512 pixels.
N = 512
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Create a pretend sample to look at. We use a simple Gaussian blob here 
# because it looks a lot like a transparent biological cell under a microscope.
true_phase = 2.0 * np.exp(-10 * (X**2 + Y**2))
amplitude  = np.ones_like(true_phase)

# Create the reference laser beam. 
# In the lab, this beam hits the camera at a slight angle.
# The kx and ky values control how tilted the beam is. 
# This tilt creates the stripe pattern (fringes) and pushes our useful data 
# away from the messy center in the Fourier transform step.
kx, ky = 20, 20
reference   = np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
object_wave = amplitude * np.exp(1j * true_phase)

# This simulates what the camera actually sees: the interference between 
# the flat reference beam and the beam that passed through our cell.
hologram = np.abs(reference + object_wave)**2

# =============================================================================
# Step 2: 2D Fast Fourier Transform
# =============================================================================
# fft2 converts our image from regular space into frequency space.
# fftshift just reorganizes the result so the lowest frequency (the bright center) 
# is right in the middle of the picture.
H = np.fft.fftshift(np.fft.fft2(hologram))

# =============================================================================
# Step 3: Locate and isolate the +1 sideband
# =============================================================================
# We take the absolute value so we can search for the brightest spots.
H_mag = np.abs(H).copy()

# The center of the image (DC term) is extremely bright. If we don't cover it up,
# our code will just select the center instead of the sideband we actually want.
# Here we put a black box over the center to hide it.
dc_block = 15
cy_centre, cx_centre = N // 2, N // 2
H_mag[cy_centre - dc_block : cy_centre + dc_block,
      cx_centre - dc_block : cx_centre + dc_block] = 0

# argmax finds the brightest single pixel left in the image, which is our sideband.
# unravel_index turns that single number into an (x, y) pixel coordinate.
cy, cx = np.unravel_index(np.argmax(H_mag), H_mag.shape)
print(f"Sideband located at pixel: row={cy}, col={cx}  "
      f"(DC centre is at {cy_centre}, {cx_centre}; "
      f"offset = {cy - cy_centre}, {cx - cx_centre})")

# Now we draw a circular mask around that bright sideband. 
# Think of it like a cookie cutter that keeps only the data inside a 30-pixel radius.
radius = 30
rows_idx = np.arange(N)[:, None]
cols_idx = np.arange(N)[None, :]
mask = (rows_idx - cy)**2 + (cols_idx - cx)**2 <= radius**2

# Multiply by the mask to delete everything outside the circle.
H_filtered = H * mask

# =============================================================================
# Step 4: Center sideband at DC, then inverse FFT
# =============================================================================
# np.roll shifts the image pixels. We are sliding our isolated sideband 
# from its corner position right back into the dead center of the image.
# If we skip this, our final picture will have a massive diagonal gradient 
# covering up the cell.
H_centred = np.roll(np.roll(H_filtered, N // 2 - cy, axis=0), N // 2 - cx, axis=1)

# Now we reverse the Fourier transform to go back to a normal looking image.
complex_field = np.fft.ifft2(np.fft.ifftshift(H_centred))

# =============================================================================
# Step 5: Extract and unwrap phase
# =============================================================================
# np.angle calculates the phase, but the math restricts it between -pi and +pi.
# This causes the image to look like a target with sharp, repeating rings.
wrapped_phase = np.angle(complex_field)

# unwrap_phase stitches those rings together to build a smooth, continuous 3D hill.
unwrapped_phase = unwrap_phase(wrapped_phase)

# =============================================================================
# Step 6: Reconstruction accuracy metric
# =============================================================================
# Sometimes the whole image is shifted up or down by a constant number.
# Subtracting the mean forces both our reconstructed image and the original 
# ground truth to start at a baseline of zero so we can compare them fairly.
phase_recon = unwrapped_phase - np.mean(unwrapped_phase)
phase_truth = true_phase      - np.mean(true_phase)

# Calculate the error map and the Root Mean Square Error (RMSE) to see how we did.
error_map = phase_recon - phase_truth
rmse      = np.sqrt(np.mean(error_map**2))
print(f"Reconstruction RMSE: {rmse:.4f} rad")

# =============================================================================
# Step 7: Visualise
# =============================================================================
# We crop the hologram panel just to zoom in and see the stripes clearly.
crop = slice(200, 300)   

# Set up a grid of 6 plots.
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Off-Axis Holographic Phase Retrieval and Simulation Pipeline", fontsize=14)

axes[0, 0].imshow(hologram[crop, crop], cmap='gray')
axes[0, 0].set_title('1. Synthetic Hologram (Zoomed)')

axes[0, 1].imshow(np.log(1 + np.abs(H)), cmap='gray')
axes[0, 1].set_title('2. FFT Spectrum\n(DC + two sidebands visible)')

axes[0, 2].imshow(true_phase, cmap='viridis')
axes[0, 2].set_title('3. True Phase; Ground Truth (rad)')

axes[1, 0].imshow(wrapped_phase, cmap='hsv')
axes[1, 0].set_title('4. Wrapped Phase (-π to +π)')

axes[1, 1].imshow(unwrapped_phase, cmap='viridis')
axes[1, 1].set_title('5. Unwrapped Phase: Reconstructed (rad)')

# Plot the error map with a custom red-to-blue color scale.
im = axes[1, 2].imshow(error_map, cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[1, 2].set_title(f'6. Reconstruction Error Map\n(RMSE = {rmse:.4f} rad)')
fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

# Turn off the axis tick marks for cleaner images.
for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('pipeline_visualisation.png', dpi=150, bbox_inches='tight')
print("Pipeline complete. Image saved as pipeline_visualisation.png")
