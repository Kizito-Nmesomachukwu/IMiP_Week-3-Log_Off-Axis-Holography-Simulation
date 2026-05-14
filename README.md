# Off-Axis Holographic Phase Retrieval Pipeline

Extract quantitative phase data from off-axis digital holograms. It is developed as part of the **3D Refractive Index Tomography** project for the *Innovation Methods in Photonics 2026* course.

## Overview 
In off-axis Mach-Zehnder interferometry, transparent biological samples (like single cells or plankton) can be visualized by measuring how they alter the phase of light. This script serves as the pipeline for the post-processing step after capturing the image from the camera.

## Pipeline

1. **Load Hologram** — load a PNG interference pattern as a grayscale intensity image
2. **FFT** — 2D Fourier transform into frequency space
3. **Sideband Isolation** — block DC, locate the +1 sideband, apply circular mask
4. **Inverse FFT** — shift sideband to center and transform back
5. **Phase Unwrapping** — convert the wrapped phase map ($-\pi$ to $+\pi$) into a continuous optical path difference map.
<!-- 6. **Background Tilt Removal** — fit and subtract a linear plane to remove residual fringe artifacts -->

## Usage

See `notebook.ipynb` for an interactive walkthrough with per-stage plots.

## Files

| File | Description |
|------|-------------|
| `Phase_Simulation.py` | Core pipeline functions and per-stage plotting |
| `notebook.ipynb` | Interactive walkthrough for debugging |
| `img/` | Input hologram images |

## Dependencies
To run this code, you will need Python 3 installed along with the following standard scientific libraries:
- `numpy` (for matrix operations and FFTs)
- `matplotlib` (for visualization)
- `scikit-image` (for the phase unwrapping algorithm)
- `Pillow` (for processing the input image)

You can install the dependencies via pip:
```bash
pip install numpy matplotlib scikit-image Pillow
