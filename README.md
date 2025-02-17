# Bob-ombs, Boos, and Toads Detection 🎮👾

## Overview 🔍

This Python script is designed to detect specific objects (Bob-ombs 💣, Boos 👻, and Toads 🍄) in a dataset of images. Using OpenCV and other image processing techniques, the script identifies and counts these objects within the images, comparing the predicted count with the ground truth provided in a CSV file. The output is evaluated using **Mean Absolute Error (MAE)** 📉, which measures the discrepancy between the predicted and actual counts.

### Key Features ✨:
- **Object Detection**: Identifies and counts Bob-ombs, Boos, and Toads in the images.
- **Preprocessing**: Enhances the image by removing green areas 🍃, sharpening 🔪, and increasing contrast 🌟.
- **Watershed Segmentation**: Applies the watershed algorithm 🌊 to detect and count individual objects.
- **Ground Truth Comparison**: Reads object counts from a CSV file 📑 and compares the model's predictions with actual counts.
- **Error Metric**: Calculates the **Mean Absolute Error (MAE)** between the predicted and true counts of objects.

---

## Requirements 📦

The following Python libraries are required to run the script:

- `numpy` 📊
- `opencv-python` (cv2) 📷
- `matplotlib` 📈
- `pandas` 🗃️
- `sklearn` 🔧

You can install the necessary libraries by running the following command:

```bash
pip install numpy opencv-python matplotlib pandas scikit-learn
```
---
### Functions 🔧
- **load_image(path)**: Loads an image from the specified path and converts it from BGR to RGB format.
- **image_gray(image)**: Converts the input image to grayscale.
- **image_bin(image_gs)**: Converts the grayscale image into a binary (black and white) image using a threshold.
- **crop_image(image, x_start, x_end, y_start, y_end)**: Crops the image to the specified coordinates.
- **display_image(image, color=False)**: Displays the image using matplotlib, with optional grayscale display.
- **sharpen_image(image)**: Applies a sharpening filter to the image.
- **remove_green(image)**: Removes green areas from the image using color thresholding in the HSV color space.
- **increase_contrast(image, alpha=1.2, beta=0)**: Increases the contrast of the image using alpha and beta parameters.
- **ws(img, i):** Applies the watershed algorithm for segmentation 🌊, detects the contours of the objects, and returns the number of objects found.
- **main(dataset_folder):** Main function that processes each image in the dataset, compares predicted and actual counts, and computes the MAE.
---

## Usage 🏃‍♂️

To run the script, use the following command:

```bash
python script_name.py path/to/dataset_folder
```
---

## Output 📊
The script will print the Mean Absolute Error (MAE) between the predicted counts and the actual counts of Bob-ombs 💣, Boos 👻, and Toads 🍄 across all images in the dataset.
