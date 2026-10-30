
# Chapter 14: Convolution
# Step 2: Create a 5x5 Image

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Create a 5x5 black-and-white image
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

# Print the image
print("--- ORIGINAL IMAGE ---")
print(image)

# Confirm image dimensions
print("\nImage Shape:", image.shape)
print("Image Size:", image.size)

print("\n5x5 image created successfully!")



# STEP 3: CREATE A HORIZONTAL EDGE DETECTOR

horizontal_filter = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

print("\n--- HORIZONTAL EDGE FILTER ---")
print(horizontal_filter)

print("\nFilter Shape:", horizontal_filter.shape)


# STEP 4: APPLY CONVOLUTION

# Apply the filter to the image
feature_map = convolve2d(
    image,
    horizontal_filter,
    mode="valid"
)

print("\n--- FEATURE MAP ---")
print(feature_map)

print("\nFeature Map Shape:", feature_map.shape)

# Find strongest positive and negative responses
print("\nHighest Value:", np.max(feature_map))
print("Lowest Value:", np.min(feature_map))

print("\nConvolution completed successfully!")



# STEP 6: VISUALIZE THE IMAGE AND FEATURE MAP

from pathlib import Path

folder = Path(__file__).resolve().parent

# Graph 1: Original image
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray", interpolation="nearest")
plt.title("Original 5x5 Plus Sign Image")
plt.xticks(range(5))
plt.yticks(range(5))
plt.colorbar(label="Pixel Value")
plt.tight_layout()

plt.savefig(folder / "Original_Plus_Sign.png", dpi=300)
plt.close()

print("\nOriginal image graph saved successfully!")

# Graph 2: Convolution feature map
plt.figure(figsize=(6, 6))
plt.imshow(feature_map, cmap="coolwarm", interpolation="nearest",
           vmin=-2, vmax=2)
plt.title("Horizontal Edge Detection Feature Map")
plt.xticks(range(3))
plt.yticks(range(3))
plt.colorbar(label="Filter Response")
plt.tight_layout()

plt.savefig(folder / "Horizontal_Edge_Feature_Map.png", dpi=300)
plt.close()

print("Feature map graph saved successfully!")
print("\nStep 6 visualization completed!")



# STEP 6: VERTICAL EDGE DETECTION

vertical_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

print("\n--- VERTICAL EDGE FILTER ---")
print(vertical_filter)

# Apply vertical filter
vertical_feature_map = convolve2d(
    image,
    vertical_filter,
    mode="valid"
)

print("\n--- VERTICAL FEATURE MAP ---")
print(vertical_feature_map)

print("\nHighest Value:", np.max(vertical_feature_map))
print("Lowest Value:", np.min(vertical_feature_map))

# Save vertical feature map visualization
plt.figure(figsize=(6, 6))

plt.imshow(
    vertical_feature_map,
    cmap="coolwarm",
    interpolation="nearest",
    vmin=-2,
    vmax=2
)

plt.title("Vertical Edge Detection Feature Map")
plt.xticks(range(3))
plt.yticks(range(3))
plt.colorbar(label="Filter Response")
plt.tight_layout()

plt.savefig(folder / "Vertical_Edge_Feature_Map.png", dpi=300)
plt.close()

print("\nVertical feature map graph saved successfully!")
print("Vertical edge detection completed!")
