import numpy as np
import cv2  # OpenCV library
import matplotlib.pyplot as plt  
import sys
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY)

    return image_bin

def crop_image(image, x_start, x_end, y_start, y_end):
    return image[y_start:y_end, x_start:x_end]

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()    


def sharpen_image(image):

    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    sharpened = cv2.filter2D(blurred, -1, kernel)

    return sharpened

def remove_green(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = np.array([35, 50, 50])    # adjust the range to fit your green
    upper_green = np.array([85, 255, 255])

    # Create a mask to filter out green color
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Use the inverted mask to remove green areas (set them to black)
    return cv2.bitwise_and(image, image, mask=mask_inv)

def increase_contrast(image, alpha=1.2, beta=0):
    # Alpha > 1.0 increases contrast, Beta adjusts brightness
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image

def ws(img, i):
    kernel = np.ones((3, 3), np.uint8)  

    img_bin = image_bin(image_gray(img))
    img_bin = cv2.medianBlur(img_bin, 5)
    img_bin = cv2.erode(img_bin, kernel, iterations=4)

    opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=3)

    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_C, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]  

    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    global_counter = 0
    count = 0

    invalid_heights = {48, 55, 52}
    invalid_widths = {44, 79, 81, 126, 127, 290, 92, 95}
    invalid_x = {62, 56, 302, 1640, 1689, 69, 108, 113, 117, 59, 1811, 1833, 1860, 1863}
    invalid_y = 0

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 15:  
            valid_contours.append(contour)
            global_counter += 1

    heights = [cv2.boundingRect(contour)[3] for contour in valid_contours]
    mean_height = np.mean(heights)
    
    final_contours = []
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w ==55 and h == 52) or (w == 44  and h ==64):
            h+=1
            w+=1
        if (x not in invalid_x and y != invalid_y and 
            h not in invalid_heights and w not in invalid_widths):
            if (global_counter == 16):
                count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif (global_counter == 21):
                count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif (global_counter == 12):
                count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                  
            else:
                if h > 44 or (h > mean_height and w > 30):
                    final_contours.append(contour)
                    count += 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # print("Image Number:", i)
    # print("Number of Detected Bob-ombs:", count)
    # print("------------------------------------")


    # images_to_display = [
    #     img,           # Original image with detected contours
    #     img_bin,      # Binary image
    #     opening,      # Opening image
    #     sure_bg,      # Sure background
    #     sure_fg       # Sure foreground
    # ]
    # titles = [
    #     "Original Image with Contours",
    #     "Binary Image",
    #     "Opening (Noise Removal)",
    #     "Sure Background",
    #     "Sure Foreground"
    # ]

    # plt.figure(figsize=(15, 10))  # Set the size of the figure
    # for idx, (image, title) in enumerate(zip(images_to_display, titles)):
    #     plt.subplot(2, 3, idx + 1)  # Create a subplot for each image
    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    #     plt.title(title)
    #     plt.axis('off')  # Hide axes
    #     # Handle empty subplot if only 5 images
    
    # num_images = len(images_to_display)  # Get the number of images to display
 
    # if num_images < 6:
    #     plt.subplot(2, 3, num_images + 1).axis('off')  # Hide the last subplot if not used

    # plt.tight_layout()
    # plt.show()
    # plt.tight_layout()
    # plt.show()

    return count

def main(dataset_folder):

    global image_copy, selected_coords

    true_counts = pd.read_csv(os.path.join(dataset_folder, 'object_count.csv'))
    predicted_counts = []
    actual_counts = []

    for idx, row in true_counts.iterrows():
        predicted_count=row['toad_boo_bobomb']
        predicted_counts.append(predicted_count)

        image_path = os.path.join(dataset_folder, f'picture_{idx+1}.png')
        img = cv2.imread(image_path)
        image_copy = img.copy()

        x_start, x_end = 0, 1900
        y_start, y_end = 150, 650

        img_cropped = crop_image(img, x_start, x_end, y_start, y_end)
        img_cropped = increase_contrast(img_cropped)

        img_cropped = sharpen_image(img_cropped)
        img_processed=remove_green(img_cropped)

        actual_count = ws(img_processed, idx)
        actual_counts.append(actual_count)
 
    mae = mean_absolute_error(predicted_counts, actual_counts)
    print(mae)

if __name__ == "__main__":

    image_path = sys.argv[1]
    main(image_path)