import pandas as pd
import math
import cv2
import os
import sys
directory = sys.argv[1]
'''
same execution style as main.py
'''
image_list = []
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.endswith('.jpg'):
        # Load the image
        image_list.append(filename)


scores_df = pd.DataFrame(columns=['image', 'score'])
os_df = pd.read_excel('os.xlsx', skiprows=[0])
od_df = pd.read_excel('od.xlsx', skiprows=[0])


def calculate_score(img_name, img):
    img_ID = '#' + img_name[8:11]
    img_side = img_name[11:13]

    if img_side == 'OS':
        row = os_df[os_df['ID'] == img_ID]
    else:
        row = od_df[od_df['ID'] == img_ID]
    axial_length = row.at[row.index[0], 'Axial_Length']

    # half so we can use  this to go either sides of the center of the image.
    if pd.isna(axial_length):
        axial_length = 26
    crop_size = (axial_length * 256) // 104

    max = int(128 + math.floor(crop_size))
    min = int(128 - math.floor(crop_size))
    img = img[min:max, min:max]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
    path = os.path.join(directory, i)
    scores_df = scores_df._append(calculate_score(
        i, cv2.imread(path)), ignore_index=True)

# write score_df to computer
scores_df.to_csv('output.csv', index=False)
