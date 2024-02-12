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
hdulist2 = fits.HDUList([hdu])

# Define the name of the FITS file
filename = 'no_blooming_bg.fits'

# Write the HDUList to a new FITS file
hdulist2.writeto(filename, overwrite=True)
#this plot looks pretty good


# %%


# PHOTOMETRY SECTION BEGINS HERE

def extractor(matrix, indices):


    # This for query-generated indices
    # the indices is usually a n x n matrix which is Boolean

    extracted_values = np.zeros_like(matrix)
    extracted_values[indices] = matrix[indices]

    return extracted_values

def extractorNonBool(matrix, indices):
    
    # the same but for n x 2 arrays with x and y indices for each column
    extracted_values = np.zeros_like(matrix)
    
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    extracted_values[row_indices, col_indices] = matrix[row_indices, col_indices]
    
    return extracted_values

def extractorBackground(matrix, indices, setValue):

    # Some issues when implementing for background, where the matrix was
    # getting wrongly updated
    matOut = np.zeros_like(matrix)
    matOut += matrix
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    # to set all the hot-pixels to the mean background radiation value 
    # of 3409 AKA mu
    matOut[row_indices, col_indices] = setValue
    
    return matOut

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

def FinalClearance(matrix, threshold, threshold2):
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
    features_to_remove = np.append(features_to_remove, np.where(max_values < threshold2))

    # Remove features with high maximum value from the matrix
    for feature_label in features_to_remove:
        matrix[labeled_features == (feature_label + 1)] = 0  # Set feature pixels to 0

    labeled_remaining = measure.label(matrix != 0)
    remaining_indices = np.argwhere(labeled_remaining)

    return matrix, remaining_indices

def combinedRemove(binaryArray, matrix, upperBound, lowerBound):
    # Apply FinalClearance on the matrix

    cleaned_matrix, posIn = lonelyRemove(binaryArray)

    inputVal = extractorNonBool(matrix, posIn)

    final_matrix, non_lone_positions = FinalClearance(inputVal, threshold=upperBound, threshold2=lowerBound)

    # Remove lonely elements from the cleaned matrix

    return cleaned_matrix, final_matrix, non_lone_positions


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

def galaxyNum(matrix):
    # Label connected components of features
    labeled_features = measure.label(matrix != 0)

    # Count the number of unique labels (excluding 0, which represents the background)
    num_features = len(np.unique(labeled_features)) - 1

    return num_features



 #%%


# lets streamline this whole process for some greater clarity in how we
# arrive at the final result, and then quickly put it all into one
# function to simplify the whole process

noBloomBack = fits.open('no_blooming_bg.fits')
noBloomValues = noBloomBack[0].data
reducedImage  = noBloomValues

def SequentialImageProcessor(data, backgroundBounds, starBounds, dustBounds, 
                             meanBackground, annulusWidth, filename):
    galaxyIndex = data > backgroundBounds
    booleanGalaxy = np.array(galaxyIndex).astype(int)

    backgroundIndex = data <= backgroundBounds
    booleanBackground = np.array(backgroundIndex).astype(int)

    galaxies = extractor(data, galaxyIndex)
    background = extractor(data, backgroundIndex)

    cleanBinaryGalaxies, cleanCountGalaxy, ObjectPositions = \
    combinedRemove(booleanGalaxy, galaxies, starBounds, dustBounds)

    inverted = np.logical_not(cleanBinaryGalaxies)
    hotPixels = np.argwhere((inverted & (booleanBackground == 0)))

    cleanBackground = extractorBackground(background, hotPixels, meanBackground)

    countAdjustment = medianBack(cleanBackground, feature_value=0, annulus_width=annulusWidth)

    adjustedGalaxyMagnitudes = magDetermine(cleanCountGalaxy, countAdjustment, 
                                            backgroundBounds)
    
    endItAll = np.copy(adjustedGalaxyMagnitudes)

    mags = fits.PrimaryHDU(endItAll)

    magList = fits.HDUList([mags])

    name = f'{filename}.fits'

    magList.writeto(name, overwrite=True)

    gCount = galaxyNum(endItAll)

    print(gCount)

    return endItAll, gCount, ObjectPositions


# %%

galaxyMags, totalCount, posGalaxy = SequentialImageProcessor(no_blooming_bg, mu + 3 * std, mu + 3 * std + 1000, mu + 3 * std + 30,  mu, 12, 'TestPLS')

# %%

plt.imshow(galaxyMags)
plt.title('Processed Image')
plt.colorbar(label = 'Counts')

 #%%

with fits.open('mosaic.fits') as hdul:
    # Get the header of the primary HDU (Header/Data Unit)
    header = hdul[0].header

zeropoint  = header['MAGZPT']
errorZeroPoint = header['MAGZRR']

def magEq(counts):
    if counts == 0:
        return 0
     
    else:
        return 25 - 2.5 * np.log10(counts)

vectorFunc = np.vectorize(magEq)

magCalibrated = vectorFunc(galaxyMags)

plt.imshow(magCalibrated)
plt.colorbar(label = 'Flux')

 #%%
