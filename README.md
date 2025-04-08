# ASG-lighting
This code extracts parameters for the Anisotropic Spherical Gaussian model to represent environmental lighting. It includes tools for fitting lobes (SG and ASG variants) to a panoramic HDR image and generates directional lighting representations.
![_comparison_3](https://github.com/user-attachments/assets/94163aaa-232e-4aaf-b3d2-82c698941944)

From left to right:
- Input HDR image
- SG map
- ASG map

## 🛠️ Installation: Required Python Libraries

To run this project, install the following libraries:

```bash
python -m pip install -U scikit-image
pip install openexr-python
pip install opencv-python
```
## 📁 Project Directory & Output Structure

- **`hdr_image/`**  
  Folder containing the source HDR environment images.

- **`output/`**  
  Directory where extracted **ASG parameters** are saved as `.npy` (NumPy array) files.

- **`output_vis/`**  
  Folder containing the generated **ASG lighting maps**, including:
  - The full-resolution **HDR map**.
  - A **tonemapped version** for visualization.

## 🧠  Configuration Parameters:

| Model Type | Parameter Index |
|------------|-----------------|
| SG         | `GAUSS_SPHERE_SX = [0,1]` |
| ASG        | `GAUSS_SPHERE_SX_SY = [1,2]` |
| ASG + Rotation. | `GAUSS_SPHERE_SX_SY_T = [2,3]` |

### `useAmbientValues` (default: `True`)
- When set to `True`, ambient lighting values are added as a background, providing a soft global illumination effect.
- When set to `False`, the background remains black.

---

### `useGoodGuess` (default: `False`)
- Enables a **pre-fitting refinement step** to improve parameter initialization.
- When `True`, the algorithm performs a more targeted analysis before fitting, allowing for more targeted and precise optimization. .
This strategy will significantly improve the initialization of parameters for fitting lighting models. The clusteringMethod can be set to one of the following options: "curveClustering", "curveClusteringParallel", or "euclidean", each representing a different approach to clustering sampled lighting directions.
The clustering process begins by generating a set of samples biased toward brighter regions in the HDR image, focusing on areas with higher light intensity. Each sample is then assigned to a detected light source using the selected clustering method—either through curve similarity, parallelized curve clustering for performance, or standard Euclidean distance. A distance metric guides the association of samples to clusters. Following clustering, Principal Component Analysis (PCA) is applied to each cluster to determine its directional orientation and intensity distribution. This analysis produces robust initial guesses for ASG parameter fitting, expecting to improve both the accuracy and convergence speed of the optimization process.
---

### `clusteringMethod`
Choose from the following clustering strategies:
- `"curveClustering"`  
- `"curveClusteringParallel"`  
- `"euclidean"`

---

## 🔬 Others 

   - `width`: Image width (default 256px).
   - `nStdAboveMean`: Light detection threshold (default 2.0).
   - `rough_level`: Surface roughness for lighting blur.

## Functionality Workflow
1. **Input Preprocessing**:
   - Normalize and resize HDR maps.
   - Enhance visibility through contrast thresholds.

2. **Light Detection**:
   - Identify local maxima in HDR maps for light-source candidates.

3. **ASG Parameter Estimation**:
   - Fit Gaussian models for detected lights (with optional anisotropy and rotation).

4. **Output ASG Map**:
   - Save HDR `.exr` image and parameters for further use.
  
## Usage
** Run the script **:
   ```bash
   python main.py
   ```


## 🧩 Code Modules Overview

---

- **`geometry.py`**  
  Handles spatial calculations and coordinate transformations, ensuring accurate representation of directions and geometry-related operations.

- **`util.py`**  
  Contains general-purpose utility functions to support image processing, normalization, file I/O, and other foundational tasks.

- **`asg_extraction.py`**  
  Implements the core logic for extracting light features from HDR images using sampling strategies and clustering-based fitting.

- **`clustering.py`**  
  Performs data clustering to group lighting samples based on intensity or spatial similarity, aiding in the initialization of ASG fitting.

- **`main.py`**  
  Serves as the entry point of the pipeline, orchestrating the full workflow from data loading and clustering to ASG parameter extraction and output generation.

---
