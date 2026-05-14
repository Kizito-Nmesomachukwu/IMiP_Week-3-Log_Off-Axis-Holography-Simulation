# Off-Axis Holographic Phase Retrieval Pipeline

This repository contains a standalone Python simulation for extracting quantitative phase data from off-axis digital holograms. It was developed as part of the **3D Refractive Index Tomography** project for the *Innovation Methods in Photonics 2026* framework.

## Overview
In off-axis Mach-Zehnder interferometry, transparent biological samples (like single cells or plankton) can be visualized by measuring how they alter the phase of light. This script simulates that entire signal-processing pipeline without needing physical hardware. 

It demonstrates:
1. **Hologram Generation:** Creating a synthetic interference fringe pattern (off-axis carrier).
2. **Fourier Filtering:** Using a 2D Fast Fourier Transform (FFT) to isolate the +1 diffraction order (sideband) while blocking the DC term.
3. **Carrier Removal:** Centering the sideband to remove the linear phase tilt.
4. **Phase Unwrapping:** Converting the wrapped phase map ($-\pi$ to $+\pi$) into a continuous optical path difference map.
5. **Error Analysis:** Comparing the reconstructed phase against the ground truth to calculate the Root Mean Square Error (RMSE).

## Dependencies
To run this code, you will need Python 3 installed along with the following standard scientific libraries:
- `numpy` (for matrix operations and FFTs)
- `matplotlib` (for visualization)
- `scikit-image` (for the phase unwrapping algorithm)

You can install the dependencies via pip:
```bash
pip install numpy matplotlib scikit-image
