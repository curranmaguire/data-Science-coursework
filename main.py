import sys
import os

import pandas as pd
import cv2
import shutil
import numpy as np
# =-------------------paths------------------
# this needs to be sys.argv[1] to make sure the passed directoresults_pathdirectory = 'test_images'
# make sure to add the sys.argv[1] + to create the results in the test images section
results_path = 'Results'
directory = sys.argv[1]
# ---------------results directory creation-------------
# Check if the directory exists
if os.path.isdir(results_path):
    # If it exists, delete its contents
    for filename in os.listdir(results_path):
        file_path = os.path.join(results_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    # If it does not exist, create the directory
    os.mkdir(results_path)

# ------------------get all images from directories names
image_list = []
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.endswith('.jpg'):
        # Load the image
        image_list.append(filename)


def show_image(img, name=''):
    cv2.imshow(name, img)

    # Display the result
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------balance the colour channels------------------

def balance_colour_channels(img):
    channels = cv2.split(img)
    k = 5
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(k,k))
    equalized_channels = [clahe.apply(channel) for channel in channels]
    #show_image(img)
    merged = cv2.merge(equalized_channels)
    #show_image(merged)
    #merged = cv2.medianBlur(merged,3)
    return merged

# -------------------change the perspective of the image using the corners


def change_perspectives(img):
    
    pts_src = np.array(
        [[60, 12], [184, 7], [201, 234], [81, 238]], dtype='float32')
    pts_dst = np.array(
        [[55, 1], [203, 2], [203, 255], [55, 254]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    img_output = cv2.warpPerspective(img, matrix, (600, 600))
    return img_output[0:256, 0:256]

# ------------------------fill in the missing region


def minor_inpaints(img):
    # this will deal with the small missing items
    show_image(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split LAB image into its channels
    h, s, v = cv2.split(hsv)

    # Apply thresholding on L channel
    thresh_value = 190
    _, mask = cv2.threshold(v, thresh_value, 240, cv2.THRESH_BINARY)
   
    inpaint = cv2.medianBlur(img, 13)
    inpaint = cv2.GaussianBlur(inpaint, (5, 5), 0)
    img[mask < 1] = inpaint[mask < 1]
    show_image(img)
    return img


def major_inpaint(img):
    # Create a mask of the missing region
    # this will be in a similar place for all images so we provide a general area

    # Set center and radius of circle
    x, y = 187, 209
    r = 24
    # Create a black image of same size as original image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Draw circle on mask
    cv2.circle(mask, (x, y), r, img.shape, -1)
   
    # Use the Navier-Stokes based inpainting algorithm
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
  
    # Use the fast marching method to inpaint small details
    fm_mask = cv2.erode(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)))
    fm_inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
  

# compared the results using this weighting and found that the standard inpaint was better
    alpha = 1
    beta = 1 - alpha
    final_inpaint = cv2.addWeighted(inpaint, alpha, fm_inpaint, beta, 0)
    # Apply post-processing
    filtered_roi = cv2.fastNlMeansDenoisingColored(
        final_inpaint, None, 6, 6, 7, 30)
    # apply the filtered inpaint to the inpaint area only
    final_inpaint[mask > 0] = filtered_roi[mask > 0]
    return final_inpaint


def fill_in_missing_area(img):

    large_inpaint = major_inpaint(img)

    final_inpaint = minor_inpaints(large_inpaint)

    return final_inpaint


# ---------------------noise and filtering


def filter_noise(img):
  
    filtered = cv2.fastNlMeansDenoisingColored(img, None, 1, 5, 10, 15)
    #show_image(filtered)
    filtered = cv2.medianBlur(filtered, 3)
    #show_image(filtered)
    
    return filtered


# ------------------------adjust the birghtness and contrast of the image


def brightness_contrast_adjust(img):
    gridsize = 4
    # Convert BGR image to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB image into its channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(gridsize, gridsize))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel with the original A and B channels
    lab_cl = cv2.merge((cl, a, b))

    # Convert LAB image back to BGR
    output_img = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)

    # show_image(img)
    # show_image(output_img)
    return output_img

    # -------------------main pipeline for image processing


# --------------------------sharpen the image-----------


def sharpen_image(img):
    sharpening_weight = 0.27
    # Convert the image from BGR to YCrCb color space
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

  
    y, cr, cb = cv2.split(img_ycrcb)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    y_sharpened = cv2.filter2D(y, -1, kernel)

    y_blended = cv2.addWeighted(
        y, 1 - sharpening_weight, y_sharpened, sharpening_weight, 0)

    # Merge the sharpened Y channel with the Cr and Cb channels
    img_ycrcb_sharpened = cv2.merge([y_blended, cr, cb])

    # Convert the sharpened image back to BGR color space
    img_bgr_sharpened = cv2.cvtColor(img_ycrcb_sharpened, cv2.COLOR_YCrCb2BGR)
    # show_image(img)
    # show_image(img_bgr_sharpened)
    return img_bgr_sharpened


# ------------------------main pipeline


def image_pipeline(img_path):
    # the main image processing loop
    img = cv2.imread(img_path)
    
    img = fill_in_missing_area(img)
    img = change_perspectives(img)
    img = filter_noise(img)
    
    img = brightness_contrast_adjust(img)
    
    img = balance_colour_channels(img)
    
    img = sharpen_image(img)

    img = cv2.resize(img, (256, 256))
    return img

scores_df = pd.DataFrame(columns=['image','score'])
os_df = pd.read_excel('os.xlsx')
od_df = pd.read_excel('od.xlsx')

def calculate_score(img_name, img):
    img_ID = '#' + img_name[8:11]
    img_side = img_name[11:13]
  
    if img_side == 'OD':
        row = os_df[os_df['ID'] == img_ID]
    else:
        row =od_df[od_df['ID'] == img_ID]
    axial_length = row.at[row.index[0], 'Axial_Length']
    
    crop_size = (axial_length * 256) // 104 #half so we can use  this to go either sides of the center of the image.
   
    max = int(128 + crop_size)
    min = int(128 - crop_size)
    img = img[min:max,min:max]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    # Count the number of white pixels in the image
    num_white_pixels = cv2.countNonZero(mask)

    # Calculate the total number of pixels in the image
    num_pixels = mask.shape[0] * mask.shape[1]

    # Calculate the number of black pixels
    

    # Calculate the black to white pixel ratio
    ratio = num_white_pixels / num_pixels
    new_row = pd.DataFrame({'image': [img_name], 'score': [ratio]})
    return new_row






# -----------------write the final image into the results section
for i in image_list:
    image_path = os.path.join(directory, i)
    write_path = os.path.join(results_path, i)

    cv2.imwrite(write_path, image_pipeline(image_path))
    #scores_df = scores_df._append(calculate_score(i, cv2.imread(write_path)), ignore_index = True)

#write score_df to computer

