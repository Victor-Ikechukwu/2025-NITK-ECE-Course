

---

# Medical Informatics, Radiomics, and Image Analysis for Computer-aided Diagnosis
**GIAN Program at NITK Surathkal**

> **Foreign Faculty:**  
> **Prof. Rangaraj Mandayam Rangayyan**  
> Professor Emeritus, Schulich School of Engineering, University of Calgary, Canada  

> **Simplified Course Material prepared by:**  
> **Dr. Agughasi Victor Ikechukwu**,  
> Department of CSE (Artificial Intelligence), MIT Mysore-India

**Course Date:** 21 FEB 2025, **Day 5 (Analysis of Oriented Patterns)**  
**Venue:** ECES 203 Seminar Hall, South-wing ECE  
**Course Webpage:** [https://sites.google.com/view/nitkecelecture/home](https://sites.google.com/view/nitkecelecture/home)

---

## Table of Contents
1. [Foreword](#foreword)  
2. [Introduction](#introduction)  
3. [Image Representations and Preliminaries](#image-representations-and-preliminaries)  
4. [Orientation Distributions and Rose Diagrams](#orientation-distributions-and-rose-diagrams)  
5. [Statistical Measures of Orientation](#statistical-measures-of-orientation)  
6. [Directional Filtering Techniques](#directional-filtering-techniques)  
7. [Transforms for Oriented Structures](#transforms-for-oriented-structures)  
8. [Detailed Use Cases](#detailed-use-cases)  
9. [Mathematical Background and Key Equations](#mathematical-background-and-key-equations)  
10. [Implementation Examples in Python](#implementation-examples-in-python)  
11. [Challenges, Limitations, and Future Directions](#challenges-limitations-and-future-directions)  
12. [Summary and Conclusions](#summary-and-conclusions)  
13. [References and Further Reading](#references-and-further-reading)

---

## 1. Foreword

> *It is my pleasure to welcome you to this session of the GIAN Program at NITK Surathkal on “Medical Informatics, Radiomics, and Image Analysis for Computer-aided Diagnosis.” This material, focusing on “Analysis of Oriented Patterns,” illustrates how powerful and instructive it can be to apply practical, hands-on methods in image analysis and pattern recognition. By exploring real-world images—such as those from biomedical imaging, mammography, and materials science—students gain an essential understanding of how to implement and interpret key algorithms for orientation detection, texture characterization, and automated decision support. These practical experiments and exercises enhance our ability to see how theoretical concepts translate into effective tools for diagnosing diseases, studying tissue properties, and designing computer-aided systems that can elevate both research and clinical outcomes. I encourage you to engage with the code snippets, the orientation-based filters, and the transform-based strategies described in this course material. By deepening your appreciation for the interplay of science, engineering, and AI-driven methods, you will be better equipped to innovate in the exciting domains of medical image analysis and radiomics. I wish you a productive and inspiring learning experience.*  
>  
> **— Prof. Rangaraj Mandayam Rangayyan**

---

## 2. Introduction

### 2.1 Motivation for Oriented Pattern Analysis

Many images—both natural and man-made—contain structures with **directional coherence** (e.g., fibers, edges, roads, vessels, elongated ridges). Analyzing or measuring these orientations is critical for:

- **Biomedical Imaging**: Collagen fibers in ligaments, vascular networks, muscle fibers, mammography (ducts, fibroglandular tissues).  
- **Remote Sensing**: Detection of roads, buildings, farmland rows in aerial or satellite images.  
- **Materials Science**: Paper grain, textile fiber orientation, carbon fiber composites.

**Why orientation matters**:
1. **Segmentation** based on dominant orientations  
2. **Quantitative assessment** (ordered vs. disordered tissues/materials)  
3. **Early detection** of subtle distortions (e.g., architectural distortion in breast images)

### 2.2 Scope and Applications

This course covers:

- **Foundational approaches**: Fourier-domain fan filtering, gradient-based orientation histograms, Gabor filters  
- **Transform-based** methods: Hough, Radon transforms  
- **Statistical tools**: Angular moments, entropy  
- **Use cases**: Ligament healing, mammograms, microvascular images, remote sensing, textiles, etc.

---

## 3. Image Representations and Preliminaries

### 3.1 Basic Notation

Let \(f(x,y)\) be a 2D grayscale image with coordinates \((x,y)\) in the range \([0,M)\times[0,N)\). The intensity may be integer (8-bit) or floating-point (normalized).

### 3.2 Local Orientation

Gradients:
\[
G_x = \frac{\partial f}{\partial x}, \quad
G_y = \frac{\partial f}{\partial y}.
\]

Local orientation angle:
\[
\theta = \mathrm{atan2}(G_y,\; G_x).
\]
We often map angles to \([0,\pi)\) or \([0^\circ,180^\circ)\) for orientation analysis.

### 3.3 Spatial vs. Frequency Domains

A line at angle \(\phi\) in the spatial domain corresponds to a perpendicular orientation in the frequency domain (often a sinc-like pattern). Fourier-based filters can isolate or enhance specific orientations.

---

## 4. Orientation Distributions and Rose Diagrams

### 4.1 Concept of Orientation and Direction Binning

A **global orientation histogram** is built by collecting local angles \(\theta\). If orientation is strongly aligned, one bin dominates.

### 4.2 Constructing a Rose Diagram

A **rose diagram** is a polar plot where each bin’s radius corresponds to the magnitude or fraction of pixels with that orientation.

### 4.3 Example Steps

1. Compute local angles \(\theta_{ij}\).  
2. (Optional) Weight each angle by gradient magnitude \(\|\nabla f\|\).  
3. Bin angles over \([0,\pi)\).  
4. Normalize and plot radially.

---

## 5. Statistical Measures of Orientation

### 5.1 Angular Moments

$$
M_k \;=\; \sum_{n=1}^{N} [\theta(n)]^k\, p(n),
$$

where \(\theta(n)\) is the center angle for bin \(n\) and \(p(n)\) is the normalized weight. Commonly:

- **First moment**: \(M_1\) ~ mean angle  
- **Second central moment**: \(M_2\) ~ orientation spread

### 5.2 Angular Entropy

$$
H \;=\; - \sum_{n=1}^{N} p(n)\,\log_2 \bigl[p(n)\bigr].
$$

- High \(H\) ~ widely scattered directions  
- Low \(H\) ~ strongly aligned

### 5.3 Principal Axis (Spatial Moments)

From standard image moments, the principal axis \(\theta^*\) satisfies:

$$
\tan(2\,\theta^*) \;=\; \frac{2\,\mu_{11}}{\mu_{20} - \mu_{02}},
$$

where \(\mu_{pq}\) are central moments. It only captures one dominant axis.

---

## 6. Directional Filtering Techniques

### 6.1 Fourier-Domain Fan Filters

1. Take FFT of image  
2. Keep frequencies in a wedge (fan) around angle \(\theta_0\)  
3. Inverse FFT

**Artifacts**: Sharp edges cause spatial ringing, so smooth transitions (Butterworth) help.

### 6.2 Gabor Filters

A Gabor filter is a Gaussian multiplied by a sinusoid, giving local frequency + orientation selectivity. We typically build a **Gabor filter bank** at multiple scales and orientations.

---

## 7. Transforms for Oriented Structures

### 7.1 Hough Transform

Parameterize lines by \((\rho, \theta)\):
$$
\rho = x\cos(\theta) + y\sin(\theta).
$$
Accumulate votes from edge pixels. Peaks correspond to strong lines.

### 7.2 Radon Transform

Integrates intensity along radial lines at varying angles \(\theta\). Useful for tomography or line detection.

### 7.3 Hough-Radon Hybrid

Accumulate actual intensity or gradient magnitude instead of just a vote of 1. This can reduce noise and handle broad lines better.

---

## 8. Detailed Use Cases

### 8.1 Collagen Fiber Analysis in Healing Ligaments

- **Challenge**: Disorganized scar tissue post-injury  
- **Method**: Apply directional filtering, measure orientation entropy  
- **Observation**: Over time, entropy typically decreases as fibers re-align

### 8.2 Microvascular Structures

- Blood vessels in pathological tissue may be more chaotic  
- Skeletonize and measure orientation distribution to detect abnormal vascular patterns

### 8.3 Architectural Distortion in Mammograms

- Subtle swirl patterns can be caught via orientation-based filtering (e.g., Gabor)  
- Check for local orientation “vortex” or spicule-like lines

### 8.4 Other Applications

1. **Textiles**: Yarn/weave alignment, broken yarn detection.  
2. **Paper**: Fiber orientation for mechanical strength.  
3. **Aerial Photos**: Road detection, farmland row analysis.  
4. **Chest X-rays**: Alveolar or vascular pathologies.  
5. **Histopathology**: Cell infiltration patterns in tissues.

---

## 9. Mathematical Background and Key Equations

### 9.1 Spatial Moments

$$
m_{pq} \;=\; \iint x^p \, y^q \, f(x,y) \,dx\,dy,
$$

and the *principal axis* orientation is found from the second central moments.

### 9.2 Fourier Transform

$$
F(u,v) \;=\; \int_{-\infty}^{+\infty}\!\!\int_{-\infty}^{+\infty} f(x,y)\, e^{-j\,2\pi\,(u\,x + v\,y)}\,dx\,dy.
$$

### 9.3 Circular Statistics

**Circular mean**  
$$
\bar{\theta} \;=\;
\mathrm{atan2}\!\Bigl(\frac{1}{N}\sum_k \sin(\theta_k),\; \frac{1}{N}\sum_k \cos(\theta_k)\Bigr).
$$

**Circular variance** similarly derived.

---

## 10. Implementation Examples in Python

Below are Python code snippets to illustrate practical aspects of oriented pattern analysis.  

### 10.1 Gradient-Based Orientation Map

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_orientation_map(image):
    """
    Compute local orientation at each pixel via Sobel gradients.
    Returns orientation angles in degrees [0..180).
    """
    img_float = np.float32(image)
    gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)

    angles = np.arctan2(gy, gx)  # -pi..pi
    angles_deg = (np.degrees(angles) + 180) % 180  # map to [0..180)
    return angles_deg

if __name__ == "__main__":
    img = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
    angles_deg = compute_orientation_map(img)
    plt.imshow(angles_deg, cmap='jet')
    plt.colorbar(label='Orientation (degrees)')
    plt.title('Local Orientation Map')
    plt.show()
```

### 10.2 Rose Diagram Computation

```python
def rose_diagram(angles_deg, mag=None, num_bins=18):
    """
    angles_deg: array of orientation angles in degrees [0..180).
    mag: optional array for weighting by gradient magnitude.
    num_bins: number of bins for [0..180) degrees.
    Returns normalized histogram p(n).
    """
    if mag is None:
        mag = np.ones_like(angles_deg)

    ang_flat = angles_deg.ravel()
    mag_flat = mag.ravel()

    hist, bin_edges = np.histogram(
        ang_flat, bins=num_bins, range=(0,180), weights=mag_flat
    )
    hist_norm = hist / (hist.sum() + 1e-6)
    return hist_norm, bin_edges

# Example usage
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
mag = np.sqrt(gx**2 + gy**2)
angles_deg = compute_orientation_map(img)
hist_norm, bin_edges = rose_diagram(angles_deg, mag=mag, num_bins=36)

center_angles = 0.5*(bin_edges[:-1] + bin_edges[1:])
plt.bar(center_angles, hist_norm, width=5.0)
plt.title('Rose Diagram (weighted by gradient magnitude)')
plt.xlabel('Orientation (degrees)')
plt.ylabel('Normalized Weight')
plt.show()
```

### 10.3 Fan Filtering in the Fourier Domain

```python
def apply_fan_filter(image, theta_center=0, theta_width=15):
    """
    Ideal 'fan' filter around theta_center ± theta_width/2 (in degrees).
    Returns the filtered image in spatial domain.
    """
    rows, cols = image.shape
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(dft_shift, dtype=np.float32)

    tc_rad = np.radians(theta_center)
    half_w = np.radians(theta_width / 2)

    for r in range(rows):
        for c in range(cols):
            y = r - crow
            x = c - ccol
            angle = np.arctan2(y, x)
            diff = abs((angle - tc_rad + np.pi) % (2*np.pi) - np.pi)
            if diff < half_w:
                mask[r, c] = 1.0

    filtered_dft = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

if __name__ == "__main__":
    filtered_0deg = apply_fan_filter(img, theta_center=0, theta_width=30)
    plt.imshow(filtered_0deg, cmap='gray')
    plt.title('Fan Filter ~0 degrees')
    plt.show()
```

### 10.4 Gabor Filter Bank

```python
def build_gabor_kernels(num_orient=6, scales=[4,8,16], ksize=31):
    """
    Create a list of Gabor kernels with different orientations and scales.
    """
    kernels = []
    for scale in scales:
        lambd = scale
        for i in range(num_orient):
            theta = np.pi * i / num_orient
            sigma = 0.5 * lambd
            gamma = 0.5
            psi = 0
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
            )
            kernels.append((kernel, theta, scale))
    return kernels

def apply_gabor_bank(image, kernels):
    responses = []
    for (k,theta,scale) in kernels:
        resp = cv2.filter2D(image, cv2.CV_32F, k)
        responses.append(resp)
    return responses

if __name__ == "__main__":
    img_gray = cv2.imread('ligament.png', 0)
    kernels = build_gabor_kernels(num_orient=8, scales=[4,8,16], ksize=31)
    responses = apply_gabor_bank(img_gray, kernels)
    # Post-process each response for orientation/scale info
```

### 10.5 Example Workflow

1. **Load** the image (e.g. SEM of a ligament).  
2. **Preprocess** (e.g., remove noise, enhance contrast).  
3. **Apply** directional filters (Gabor or fan).  
4. **Threshold** to isolate oriented features.  
5. **Compute** a rose diagram or orientation histogram.  
6. **Derive** statistics (mean angle, second moment, entropy).  
7. **Interpret** results (e.g. high entropy → disorganized).

---

## 11. Challenges, Limitations, and Future Directions

- **Artifacts** from Fourier filtering (ringing).  
- **Multi-modal orientation** may require multi-peak detection.  
- **Nonlinear transforms** (wavelets, shearlets, curvelets) for curved or complex patterns.  
- **Deep learning** integration: directional filters inside CNN layers, etc.

---

## 12. Summary and Conclusions

Oriented pattern analysis is critical for:
- **Segmenting** fibrous/elongated structures,  
- **Quantifying** alignment vs. disorder (entropy, dispersion),  
- **Detecting** subtle distortions (e.g. in mammograms, ligaments, or textiles).

**Key Approaches**:
1. Gradient-based orientation + rose diagrams  
2. Fourier fan filtering  
3. Gabor filter banks  
4. Hough/Radon transforms for line detection  

Applications span **biomedical imaging**, **remote sensing**, **materials science**, etc.

---

## 13. References and Further Reading

1. **R. M. Rangayyan**, *Biomedical Image Analysis*, CRC Press  
2. **G. Granlund and H. Knutsson**, *Signal Processing for Computer Vision*, Kluwer  
3. **M. Petrou and P. Bosdogianni**, *Image Processing: The Fundamentals*, Wiley  
4. **J. G. Daugman**, “Complete discrete 2D Gabor transforms by neural networks for image analysis and compression,” *IEEE Trans. Acoustics, Speech, Signal Processing*, 1988  
5. **X. Li**, “Hough transform for line detection: Implementation details,” *Pattern Recognition*, 1996  
6. **W. K. Pratt**, *Digital Image Processing*, Wiley  

---
