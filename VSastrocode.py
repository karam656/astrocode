#%%
import numpy as np 
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

#%%
#loading in data
hdulist = fits.open("mosaic.fits")

pixelvalues = hdulist[0].data

pixelvalues_cropped_horizontally = pixelvalues[:, 120:-120]

# Crop vertically by slicing only rows from 110 to -110, keeping all columns
pixelvalues = pixelvalues_cropped_horizontally[110:-110, :]

#%%
#plottinh histogram, mask used to remove anomolies/ clean data
mask = (3000 < pixelvalues) & (pixelvalues < 4000)

# Use the mask to select pixels within the desired range
selected_pixels = pixelvalues[mask]

def skewed_gaussian(x, amp, a, mean, std):
    return amp * skewnorm.pdf(x, a, loc=mean, scale=std)

hist, bins = np.histogram(selected_pixels, bins = 1000)

params, covariance = curve_fit(skewed_gaussian, bins[:-1], hist, p0 = (900000, 1.5, 3425, np.sqrt(3425)), maxfev = 100000)

plt.figure(figsize=(10, 10))
plt.bar(bins[:-1], hist, width = np.diff(bins), edgecolor='black', label = 'Data ')
plt.plot(bins[:-1], 1.3 * skewed_gaussian(bins[:-1], *params), label = 'Gauss fit', color = 'red')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.legend()
plt.grid()
plt.xlim(3300,3700)
plt.show()

#the mean of this histogram should be like the background noise

#%%
def masker(data, cutoff):
    max_length = max(len(sublist) for sublist in data)
    data = np.array([np.pad(sublist, (0, max_length - len(sublist)), 'constant', constant_values=0) for sublist in data])
    
    mask_arr = np.zeros(data.shape)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] >= cutoff:
                mask_arr[i][j] = 1
    return mask_arr

# Test data with varying lengths

testdata = [[0, 0, 1, 2, 5, 1.1, 0], [2, 1, 3, 0, 0, 0.9], [2, 0, 1]]


# Apply the mask function
test1 = masker(testdata, 1)


#%%

#converting to flux
#from header we see gain is 3.1, lab dude said flux = counts*gain/sattime
gain = 3.1
x_time = 720

def flux(data, x_time, gain):
    flux = data*gain/x_time
    return flux

# %%

mu = params[2]

masked = masker(pixelvalues, mu)
plt.figure(figsize = (10, 10))
plt.imshow(masked, cmap = 'gray')


# %%

import skimage as sk
from skimage import measure, morphology

#remember to import stuff from astrocode
#%%
def skimmyging(data, cutoff, area):
    pixelvalues_cropped_01 = masker(data, cutoff)
    #binning of objects of area greater than 1800
    # Step 1: Label the objects in the image
    label_image, num_features = measure.label(pixelvalues_cropped_01, background=0, return_num=True)

    # Step 2: Measure properties of labeled objects
    regions = measure.regionprops(label_image)

    # Step 3: Identify objects with 1800 or more pixels and set their pixels to 0
    for region in regions:
        if region.area >= area:
            # Set pixels of this region to 0 in the original image
            for coordinates in region.coords:
                pixelvalues_cropped_01[coordinates[0], coordinates[1]] = 0

    # If you want to create a modified image where the identified objects are removed,
    # you can use the label_image to mask the original image.
    modified_image = np.where(label_image == 0, 0, pixelvalues_cropped_01)

    plt.imshow(modified_image, cmap = 'gray')
    plt.show()

#applying skimmyging to our data with loose params
data = pixelvalues
std = params[3]
cutoffSet = mu + 3.5 * std
area = 1900

skimmyging(data, cutoffSet, area)

#%%

# I want to go back to the og data without the hot stuff
def skimmyging_with_change_indicator(data, cutoff, area):
    # Apply the mask to identify bright sources
    pixelvalues_cropped_01 = masker(data, cutoff)
    
    # Label the objects in the masked image
    label_image, num_features = measure.label(pixelvalues_cropped_01, background=0, return_num=True)
    
    # Initialize the change indicator array with zeros
    change_indicator = np.zeros(data.shape, dtype=int)
    
    # Measure properties of labeled objects
    regions = measure.regionprops(label_image)

    for region in regions:
        if region.area >= area:
            for coordinates in region.coords:
                # Set pixels of this region to 0 in the original data
                data[coordinates[0], coordinates[1]] = 0
                # Mark the change in the change indicator array
                change_indicator[coordinates[0], coordinates[1]] = 1

    # Optionally, you can visualize the change_indicator array
    plt.imshow(change_indicator, cmap='gray')
    # plt.show()

    return data, change_indicator

skim_data, change_indicator = skimmyging_with_change_indicator(pixelvalues, mu + 3*std, 1850)

#%%

def compare(data, indicator):
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if indicator[i][j]==1:
                data[i][j] = 0
    return data

no_blooming_bg = compare(pixelvalues, change_indicator)
#plotting this isn't too nice

#%%
#trying to save to a fits file to 
hdu = fits.PrimaryHDU(no_blooming_bg)

# Create a HDUList to contain the HDU
hdulist = fits.HDUList([hdu])

# Define the name of the FITS file
filename = 'no_blooming_bg.fits'

# Write the HDUList to a new FITS file
hdulist.writeto(filename, overwrite=True)
#this plot looks pretty good

# %%

# impor the data 
noBloomBack = fits.open('no_blooming_bg.fits')

noBloomValues = noBloomBack[0].data


# %%

# take a snapshot for now, for simplicity
reducedImage  = noBloomValues

# threshold to remove background, identify indicies where this is true
indicesReduce = reducedImage > mu + 3 * std

# threshold to keep background only, remove galaxies
backIndex = reducedImage <= mu + 3 * std

# %%

def extractor(matrix, indices):
    extracted_values = np.zeros_like(matrix)
    
    # Assign the values at the positions where boolean_indices is True to the corresponding positions in the new matrix
    extracted_values[indices] = matrix[indices]

    return extracted_values

def extractorNonBool(matrix, indices):
    # Create a new matrix of zeros with the same shape as the original matrix
    extracted_values = np.zeros_like(matrix)
    
    # Extract row and column indices from the 'indices' array
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    # Assign the values at the specified indices to the corresponding positions in the new matrix
    extracted_values[row_indices, col_indices] = matrix[row_indices, col_indices]
    
    return extracted_values

def extractorBackground(matrix, indices, setValue):
    # Create a new matrix of zeros with the same shape as the original matrix
    matOut = np.zeros_like(matrix)
    
    matOut += matrix
    # Extract row and column indices from the 'indices' array
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    # Assign the values at the specified indices to the corresponding positions in the new matrix
    matOut[row_indices, col_indices] = setValue
    
    return matOut

# extract the background free galaxies, non-binarised dataset
backLessGalaxy = extractor(reducedImage, indicesReduce)

#extract only background, without galaxies, non binarised dataset
backWithOnly = extractor(reducedImage, backIndex)


# %% 

boolReduce = np.array(indicesReduce).astype(int)
boolBack = np.array(backIndex).astype(int)

plt.imshow(boolBack)

# %%
def lonelyRemove(binary_array):
    # Label connected components
    labeled_array, num_features = measure.label(binary_array, connectivity=2, return_num=True)

    # Count sizes of each connected component
    sizes = np.bincount(labeled_array.ravel())

    # Get indices of components with size 1 (lone elements)
    lone_element_indices = np.where(sizes == 1)[0]

    # Remove lone elements from labeled array
    for idx in lone_element_indices:
        labeled_array[labeled_array == idx] = 0

    # Reconstruct binary array without lone elements
    filtered_binary_array = labeled_array > 0

    non_lone_positions = np.argwhere(filtered_binary_array)

    return filtered_binary_array, non_lone_positions


# %%

# positions is the indicies for objects that are not hot-pixels
filtered, positions = lonelyRemove(boolReduce)

cleanBackGalaxy = extractorNonBool(backLessGalaxy, positions)

plt.figure(figsize=(10, 10))
plt.imshow(cleanBackGalaxy)


# %%

inverted_filter= np.logical_not(filtered)

plt.imshow(inverted_filter)

zeroIndices = np.argwhere((inverted_filter & (boolBack == 0)))

# %%

cleanBackOnly = extractorBackground(backWithOnly, zeroIndices, mu)

plt.imshow(cleanBackOnly)

# %%

def medianBack(matrix, feature_value=0, annulus_width=12):
    # Label connected components of features
    labeled_features = measure.label(matrix == feature_value)

    # Calculate the median value for each feature
    medians = []
    for props in measure.regionprops(labeled_features):
        # Get the bounding box coordinates for the current feature
        min_row, min_col, max_row, max_col = props.bbox

        # Expand the bounding box to form an annulus with a width of annulus_width
        min_row = max(0, min_row - annulus_width // 2)
        max_row = min(matrix.shape[0], max_row + annulus_width // 2)
        min_col = max(0, min_col - annulus_width // 2)
        max_col = min(matrix.shape[1], max_col + annulus_width // 2)

        # Extract values within the annulus window
        annulus_values = matrix[min_row:max_row, min_col:max_col].flatten()

        nonZero = annulus_values > 0

        medianMetrics = annulus_values[nonZero]

        # Calculate the median of the annulus values and append to the list
        medians.append(np.median(medianMetrics))

    return medians

 #%%

medianBackground = medianBack(cleanBackOnly, feature_value=0, annulus_width=12)

# %%

def magDetermine(matrix, median_values, threshold):
    # Create a copy of the input matrix to avoid modifying the original data
    matrix_copy = np.copy(matrix).astype(float)

    # Label connected components of features based on the condition
    labeled_features = measure.label(matrix > threshold)

    # Subtract the corresponding median value from each feature region
    for props, median_value in zip(measure.regionprops(labeled_features), median_values):
        # Get the bounding box coordinates for the current feature
        min_row, min_col, max_row, max_col = props.bbox

        # Subtract the median value from the feature region
        matrix_copy[min_row:max_row, min_col:max_col][labeled_features[min_row:max_row, min_col:max_col] == props.label] -= median_value

    return matrix_copy

 #%%

finalMags = magDetermine(cleanBackGalaxy, medianBackground, mu + 3 * std)


# %%


def FinalClearance(matrix, threshold=1000):
    # Label connected components of features
    labeled_features = measure.label(matrix != 0)

    # Calculate maximum value within each feature
    max_values = []
    for props in measure.regionprops(labeled_features):
        max_value = np.max(matrix[props.coords[:, 0], props.coords[:, 1]])
        max_values.append(max_value)

    max_values = np.array(max_values)

    # Find features with maximum value greater than the threshold
    features_to_remove = np.where(max_values > threshold)[0]

    # Remove features with high maximum value from the matrix
    for feature_label in features_to_remove:
        matrix[labeled_features == (feature_label + 1)] = 0  # Set feature pixels to 0

    return matrix

#%%

endGoal = FinalClearance(finalMags, threshold=1000)

plt.imshow(endGoal)
# %%

#trying to save to a fits file to 
mags = fits.PrimaryHDU(endGoal)

# Create a HDUList to contain the HDU
magList = fits.HDUList([mags])

# Define the name of the FITS file
filename = 'snapshotgalaxies.fits'

# Write the HDUList to a new FITS file
magList.writeto(filename, overwrite=True)
#this plot looks pretty good

# %%

def galaxyNum(matrix):
    # Label connected components of features
    labeled_features = measure.label(matrix != 0)

    # Count the number of unique labels (excluding 0, which represents the background)
    num_features = len(np.unique(labeled_features)) - 1

    return num_features

# %%

gCount = galaxyNum(endGoal)

print(gCount)

# %%

