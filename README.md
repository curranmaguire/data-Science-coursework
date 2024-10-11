# Eye Health Analysis Project

## Overview

This project analyzes eye health metrics from two datasets: `os` (left eye) and `od` (right eye). It combines data analysis with image processing of retinal images to aid in diagnosis. The project includes data preprocessing, statistical analysis, and an image processing pipeline for enhancing retinal images.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- xlsxwriter
- opencv-python (cv2)
- sys
- os
- math
- shutil

## Data Files

- `od#.xlsx`: Dataset for right eye measurements
- `os#.xlsx`: Dataset for left eye measurements
- Retinal images in JPG format

## Project Components

### 1. Data Analysis

1. **Data Loading and Preprocessing**
   - Load the Excel files using pandas
   - Combine the datasets into a single DataFrame
   - Handle missing data and outliers
   - Encode categorical variables

2. **Data Cleaning and Transformation**
   - Check for incorrect formatting in categorical variables
   - Encode string values to numeric for analysis
   - Remove outliers using IQR and Z-score methods
   - Fill missing data using appropriate methods (mean, median, mode)

3. **Exploratory Data Analysis**
   - Generate correlation heatmaps
   - Create pairplots to visualize relationships between variables
   - Analyze the distribution of diagnoses (healthy, glaucomatous, suspicious)

4. **Statistical Analysis**
   - Calculate skewness for specific columns
   - Perform Missing Completely at Random (MCAR) test

5. **Visualizations**
   - Box plots for various metrics
   - Scatter plots for selected columns
   - Histograms for data distribution analysis

6. **Data Export**
   - Save cleaned datasets back to Excel files with formatted headers

## Key Functions

- `missing_data_counter(df)`: Calculates the percentage of missing values for each column
- `check_formatting(df, id, expected_values)`: Checks for incorrect values in specified columns
- `encode_strings(df, id, values)`: Encodes string values to numeric
- `box_plot_columns(df, ids)`: Creates box plots for specified columns
- `scatter_plot_columns(df, ids)`: Generates scatter plots for specified columns
- `histogram_for_columns(df, ids)`: Creates histograms for specified columns
- `detect_outliers(df, ids)`: Detects and removes outliers using IQR method
- `Z_score_outlier_removal(df, id)`: Removes outliers using Z-score method
- `normal_mcar_fill(df, ids)`: Fills missing values with mean for normally distributed data
- `skewed_mcar_fill(df, ids)`: Fills missing values with median for skewed data
- `mode_missing_data_fill(df, id)`: Fills missing values with mode
- `save_dataset(df, name)`: Saves the cleaned dataset to an Excel file with formatted headers

## Usage

To run the analysis:

1. Ensure all dependencies are installed
3. Run the Python script containing the analysis code in order (1,2,3)
4. The script will generate cleaned datasets (`od_cleaned.xlsx` and `os_cleaned.xlsx`) and various visualizations

## Future Improvements

- Implement machine learning models for prediction
- Add statistical tests to validate findings
- Expand the dataset with more samples or additional eye health metrics
- Develop an interactive dashboard for easier data exploration

### 2. Image Processing Pipeline

The project includes an image processing pipeline to enhance retinal images for better diagnosis. The pipeline is implemented in a separate Python script.

#### Key Functions in Image Processing:

- `balance_colour_channels(img)`: Balances color channels using CLAHE
- `change_perspectives(img)`: Corrects image perspective
- `minor_inpaints(img)`: Fills small missing areas in the image
- `major_inpaint(img)`: Fills larger missing areas using inpainting techniques
- `fill_in_missing_area(img)`: Combines major and minor inpainting
- `filter_noise(img)`: Applies noise reduction filters
- `brightness_contrast_adjust(img)`: Adjusts image brightness and contrast
- `sharpen_image(img)`: Enhances image sharpness
- `image_pipeline(img_path)`: Main function that applies all processing steps

#### Image Processing Workflow:

1. Load images from the specified directory
2. For each image:
   - Fill in missing areas
   - Correct perspective
   - Reduce noise
   - Adjust brightness and contrast
   - Balance color channels
   - Sharpen the image
   - Resize to 256x256 pixels
3. Save processed images in the 'Results' directory

## Usage

To run the image processing pipeline:

1. Ensure all dependencies are installed
2. Place the retinal images in a directory
3. Run the image processing script with
