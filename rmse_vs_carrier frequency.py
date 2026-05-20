#Carrier Frequency Sweep Simulation: Quantifying the Minimum Off-Axis Angle for Reliable Phase Reconstruction



import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

# numpy handles all the array math and the FFT operations.
# matplotlib draws all the plots and saves them to disk.
# unwrap_phase from skimage fixes the 2-pi jumps in the raw phase output.

# =============================================================================
# Title: RMSE vs Off-Axis Carrier Frequency Sweep
#        Quantifying the minimum mirror tilt needed for clean phase retrieval
# =============================================================================
# Description:
# This script answers a specific question that came out of Week 3 of the lab log:
# how large does our physical off-axis angle actually need to be for the
# phase reconstruction to work reliably?
#
# Rather than guessing in the lab, I built a simulation that sweeps through
# ten different carrier frequencies (each one corresponding to a different
# mirror tilt angle) and measures the reconstruction error at each setting.
# The error is quantified using RMSE between the reconstructed phase and the
# known ground truth of the synthetic sample.
#
# The output is a bar chart with a logarithmic y-axis. Red bars show cases
# where the carrier is too low and the sideband crashes into the DC term,
# producing massive errors. Blue bars show cases where the separation is
# clean and the reconstruction is reliable. The top axis of the chart
# converts each carrier frequency into an approximate physical mirror angle
# in degrees, so the result can be used directly to guide lab alignment.
#
# This script was written as part of Week 4 of the Innovation Methods in
# Photonics 2026 project log and builds directly on the synthetic pipeline
# in pipeline_simulation.py.

# =============================================================================
# Background
# =============================================================================
# In the lab we built a Mach-Zehnder interferometer and tried to record
# off-axis holograms of transparent samples. When we ran the first images
# through the Fourier transform we got a mess of unwanted dots instead of
# a clean sideband. We suspected the off-axis angle of our reference beam
# was too small, which causes the sideband and the DC term to sit on top
# of each other in the frequency domain. But we did not know exactly how
# small is too small.
#
# Instead of guessing in the lab, I built this simulation to sweep through
# a range of carrier frequencies (which map directly to mirror tilt angles)
# and measure the reconstruction error at each one. The output is a bar chart
# that shows clearly where the safe zone is and where things fall apart.

# =============================================================================
# Known Limitations
# =============================================================================
# The simulation assumes a perfect, noise-free hologram with uniform amplitude.
# Real holograms have camera noise, laser fluctuations, dust on optics, and
# mechanical vibrations from the table. All of those will push the RMSE higher
# than what you see in the blue bars here. Think of this result as a best-case
# lower bound. Getting below 0.5 rad on real experimental data at this stage
# of the setup would already be a reasonable result.
#
# The mask radius is fixed at 30 pixels for every carrier frequency tested.
# In practice, the right radius depends on the complexity of the object being
# imaged. A more complex object spreads the sideband lobe wider and may need
# a larger mask. This should be tuned once real data is available.

# =============================================================================
# Dependencies
# =============================================================================
# numpy, matplotlib, scikit-image
# All three come pre-installed in Google Colab. No pip install needed.

# =============================================================================
# Parameters you can change
# =============================================================================
# N            : grid size in pixels. 512 is a good balance of speed and accuracy.
# DC_BLOCK     : half-width in pixels of the box zeroed around the DC term
#                when searching for the sideband. Keep it smaller than the
#                sideband offset or you will accidentally suppress the signal.
# MASK_RADIUS  : radius in pixels of the circular filter around the sideband.
#                Too small and you clip the lobe and lose phase detail.
#                Too large and you let in noise from neighboring regions.
# WAVELENGTH   : your laser wavelength in metres. Used only for the angle
#                calculation on the top axis of the plot.
# PIXEL_SIZE   : your camera pixel size in metres. Check the datasheet.
#                Also used only for the angle calculation.
# carriers     : list of carrier frequencies to sweep over. Add or remove
#                values to cover a different range.

# =============================================================================
# Scene setup: 512x512 synthetic transparent cell with a Gaussian phase object
# =============================================================================

N        = 512
x        = np.linspace(-1, 1, N)
X, Y     = np.meshgrid(x, x)

# The fake sample is a Gaussian blob peaking at 2 radians in the centre.
# This is a reasonable first approximation for a small transparent biological cell.
# The amplitude is set to 1 everywhere because we only care about phase here.
true_phase  = 2.0 * np.exp(-10 * (X**2 + Y**2))
phase_truth = true_phase - np.mean(true_phase)

# =============================================================================
# Fixed parameters across all trials
# =============================================================================

DC_BLOCK    = 15      # pixels, half-width of the DC suppression window
MASK_RADIUS = 30      # pixels, sideband filter radius

WAVELENGTH  = 532e-9  # metres, green laser
PIXEL_SIZE  = 3.45e-6 # metres, typical CMOS sensor pixel size

# =============================================================================
# Carrier frequency sweep
# =============================================================================
# Each value of kx sets the tilt of the off-axis reference beam in the simulation.
# The pixel offset of the sideband from the DC centre equals 2 * kx.
# This comes from the coordinate system: x goes from -1 to 1, so one full
# carrier cycle spans 2 coordinate units, which the FFT resolves as a 2*kx bin shift.
#
# The sideband starts overlapping the DC term when:
#     offset - MASK_RADIUS < DC_BLOCK
#     2*kx - 30 < 15
#     kx < 22.5
#
# So anything below kx = 23 is in the danger zone for these parameter settings.

carriers     = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
rmse_list    = []
overlap_flag = []

for kx in carriers:
    ky = kx  # diagonal carrier so the sideband moves along both axes equally

    # Build the hologram for this carrier frequency.
    # The reference is a tilted plane wave. The object wave is our fake cell.
    # The camera records only the intensity: the squared magnitude of their sum.
    # Expanding that square gives four terms: two DC-like intensity terms,
    # one cross-term carrying +object_phase, and one carrying -object_phase.
    # The whole pipeline is about isolating that first cross-term.
    reference = np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    hologram  = np.abs(reference + np.exp(1j * true_phase))**2

    # Take the 2D FFT and shift so DC sits in the centre of the image.
    # In the spectrum you should see three blobs: the bright DC lump in the
    # middle and two sidebands placed symmetrically around it.
    H     = np.fft.fftshift(np.fft.fft2(hologram))
    H_mag = np.abs(H).copy()

    # Zero out the DC region so it does not dominate the sideband search.
    # Then blank the bottom half and right half of the spectrum.
    # We do this because the hologram is a real-valued image, which means
    # its spectrum is Hermitian: the two sidebands carry conjugate information.
    # The top-left sideband holds +object_phase.
    # The bottom-right sideband holds -object_phase.
    # Searching only the top-left quadrant means we always pick the right one
    # and avoid getting a sign-flipped phase reconstruction for large kx values.
    H_mag[N//2 - DC_BLOCK : N//2 + DC_BLOCK,
          N//2 - DC_BLOCK : N//2 + DC_BLOCK] = 0
    H_search           = H_mag.copy()
    H_search[N//2:, :] = 0  # blank bottom half
    H_search[:, N//2:] = 0  # blank right half
    cy, cx = np.unravel_index(np.argmax(H_search), H_mag.shape)

    # Calculate the actual pixel offset of the sideband from the DC centre.
    # Flag whether the sideband lobe overlaps the DC block region.
    # Overlap happens when the edge of the mask (offset - MASK_RADIUS)
    # is still inside the DC suppression window (less than DC_BLOCK).
    offset_px = int(np.round(np.sqrt((cy - N//2)**2 + (cx - N//2)**2)))
    overlaps  = (offset_px - MASK_RADIUS) < DC_BLOCK
    overlap_flag.append(overlaps)

    # Draw a circular mask around the sideband peak.
    # This keeps only the frequency content we want and zeros out everything else.
    rows_idx = np.arange(N)[:, None]
    cols_idx = np.arange(N)[None, :]
    mask     = (rows_idx - cy)**2 + (cols_idx - cx)**2 <= MASK_RADIUS**2
    H_filt   = H * mask

    # Roll the sideband to the centre of the frequency grid before inverting.
    # This removes the carrier tilt from the reconstruction.
    # Without it, the unwrapped phase has a strong linear ramp across the whole
    # image and the actual object phase is buried under it.
    # Rolling by (N//2 - cy) and (N//2 - cx) moves the sideband to (N//2, N//2).
    H_cent = np.roll(np.roll(H_filt, N//2 - cy, axis=0), N//2 - cx, axis=1)
    cf     = np.fft.ifft2(np.fft.ifftshift(H_cent))

    # Extract the phase. np.angle gives values in [-pi, +pi].
    # unwrap_phase removes the 2-pi jumps to give a smooth continuous surface.
    up = unwrap_phase(np.angle(cf))

    # Subtract the mean from both maps before comparing.
    # The absolute phase reference is arbitrary so a global offset can exist
    # between the reconstruction and the ground truth. Mean subtraction removes
    # it so the comparison is fair.
    phase_recon = up - np.mean(up)
    rmse        = np.sqrt(np.mean((phase_recon - phase_truth)**2))
    rmse_list.append(rmse)

    status = "OVERLAP" if overlaps else "OK"
    print(f"kx={kx:>2}  offset={offset_px:>3} px  RMSE={rmse:.4f} rad  [{status}]")

# =============================================================================
# Plot: RMSE vs carrier frequency
# =============================================================================
# Red bars are overlap cases where DC and sideband contaminate each other.
# Blue bars are clean separations where the reconstruction is reliable.
# The top x-axis converts kx into an approximate physical mirror angle using:
#     theta = arctan( wavelength / (pixel_size * pixel_offset) )
# This is the angle you would need to set on the kinematic mirror in the lab.

fig, ax1 = plt.subplots(figsize=(11, 5.5))

bar_colors = ['#d62728' if ov else '#1f77b4' for ov in overlap_flag]
bars = ax1.bar(carriers, rmse_list,
               color=bar_colors, width=1.6, alpha=0.88,
               edgecolor='white', linewidth=0.8, zorder=3)

# Write the RMSE value on top of each bar so exact numbers are readable.
for bar, rmse, ov in zip(bars, rmse_list, overlap_flag):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() * 1.05,
             f'{rmse:.2f}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold',
             color='#d62728' if ov else '#1f77b4')

# Green dashed line marks the practical acceptance threshold.
# Anything above this is too noisy for reliable phase reconstruction.
THRESHOLD = 0.35
ax1.axhline(THRESHOLD, color='#2ca02c', linestyle='--', linewidth=1.8,
            label=f'Practical threshold ({THRESHOLD} rad)', zorder=4)

# Shade the region where overlap occurs.
kx_boundary = (DC_BLOCK + MASK_RADIUS) / 2
ax1.axvspan(0, kx_boundary, color='#d62728', alpha=0.07,
            label=f'DC/sideband overlap zone  (kx < {kx_boundary:.0f})')

# Orange dotted line marks the first carrier frequency that clears the overlap zone.
min_good_kx = next(k for k, ov in zip(carriers, overlap_flag) if not ov)
ax1.axvline(min_good_kx, color='#ff7f0e', linestyle=':', linewidth=1.5,
            label=f'Minimum safe carrier  kx = {min_good_kx}')

# Log scale on y so the small blue bar values are visible next to the large red ones.
ax1.set_yscale('log')
ax1.set_xlabel('Carrier frequency  kx  [cycles per unit length]', fontsize=12)
ax1.set_ylabel('Phase reconstruction RMSE  [rad]  log scale', fontsize=12)
ax1.set_title(
    'Effect of Off-Axis Carrier Frequency on Phase Reconstruction Quality\n'
    'Red bars: DC/sideband overlap, corrupted phase  |  Blue bars: clean separation',
    fontsize=11)
ax1.set_xticks(carriers)
ax1.set_xticklabels([f'kx = {k}\n({2*k} px offset)' for k in carriers], fontsize=8)
ax1.grid(axis='y', linestyle=':', alpha=0.45, zorder=0)
ax1.legend(fontsize=9.5, loc='upper right')

# Secondary top axis showing the approximate physical off-axis angle in degrees.
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(carriers)
angles_deg = [
    np.degrees(np.arctan(WAVELENGTH / (PIXEL_SIZE * max(2 * k, 1))))
    for k in carriers
]
ax2.set_xticklabels([f'{a:.2f}°' for a in angles_deg], fontsize=8, color='#555555')
ax2.set_xlabel(
    f'Approx. physical off-axis angle  (wavelength = {int(WAVELENGTH*1e9)} nm, '
    f'pixel = {PIXEL_SIZE*1e6:.2f} um)',
    fontsize=9.5, color='#555555')

plt.tight_layout()
plt.savefig('rmse_vs_carrier.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done.")
