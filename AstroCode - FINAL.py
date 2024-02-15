#%%
import numpy as np 
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

import skimage as sk
from skimage import measure

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

gain = 3.1
x_time = 720
mu = params[2]
std = params[3]

#applying skimmyging to our data with loose params
data = pixelvalues
std = params[3]
cutoffSet = mu + 3 * std
Apass = 1800


#%%

# THIS IS CONFIRM GOOD SHIT
# I want to go back to the og data without the hot stuff
def skimmyging_with_change_indicator(data, cutoff, area):

    # Create a copy of the data to work with
    storage = np.copy(data)
    
    # Apply the mask to identify bright sources
    pixelvalues_cropped_01 = masker(storage, cutoff)
    
    # Label the objects in the masked image
    label_image, num_features = measure.label(pixelvalues_cropped_01, background=0, return_num=True)
    
    # Initialize the change indicator array with zeros
    change_indicator = np.zeros(storage.shape, dtype=int)
    
    # Measure properties of labeled objects
    regions = measure.regionprops(label_image)

    for region in regions:
        if region.area >= area:
            for coordinates in region.coords:
                # Set pixels of this region to 0 in the original data
                storage[coordinates[0], coordinates[1]] = 0
                # Mark the change in the change indicator array
                change_indicator[coordinates[0], coordinates[1]] = 1

    # Optionally, you can visualize the change_indicator array
    plt.imshow(change_indicator, cmap = 'gray')
    plt.show()

    return storage, change_indicator

skim_data, changer = skimmyging_with_change_indicator(pixelvalues, cutoffSet, 1850)

#%%

# THIS IS ALSO GOOD
def compare(data, indicator):
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if indicator[i][j]==1:
                data[i][j] = 0
    return data

no_blooming_bg = compare(pixelvalues, changer)
#plotting this isn't too nice

#%%
#trying to save to a fits file to 
# THIS IS GOOD
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

# THIS IS GOOD
def extractor(matrix, indices):

    # This for query-generated indices
    # the indices is usually a n x n matrix which is Boolean

    extracted_values = np.zeros_like(matrix)
    extracted_values[indices] = matrix[indices]

    return extracted_values

# THIS IS GOOD
def extractorNonBool(matrix, indices):
    
    # the same but for n x 2 arrays with x and y indices for each column
    extracted_values = np.zeros_like(matrix)
    
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    extracted_values[row_indices, col_indices] = matrix[row_indices, col_indices]
    
    return extracted_values

# THIS IS GOOD
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

# THIS LOOKS GOOD
def lonelyRemove(binary_array):
    # Label connected components
    labeled_array, num_features = measure.label(binary_array, connectivity=2, return_num=True)

    # Count sizes of each connected component
    sizes = np.bincount(labeled_array.ravel())

    # Get indices of components with size 1 (lone elements)
    lone_element_indices = np.where(sizes == 1)[0]

    for idx in lone_element_indices:
        labeled_array[labeled_array == idx] = 0

    filtered_binary_array = labeled_array > 0

    non_lone_positions = np.argwhere(filtered_binary_array)

    return filtered_binary_array, non_lone_positions

# IM NOT SURE ABOUT THIS FUNCTION ONLY
# THIS MIGHT HAVE SOME PROBLEMS - DOES NOT


def FinalClearance(matrix, threshold, threshold2):

    labeled_features = measure.label(matrix != 0)


    max_values = []
    for props in measure.regionprops(labeled_features):
        max_value = np.max(matrix[props.coords[:, 0], props.coords[:, 1]])
        max_values.append(max_value)

    max_values = np.array(max_values)


    features_to_remove = np.where(max_values > threshold)[0]
    features_to_remove = np.append(features_to_remove, np.where(max_values < threshold2)[0])


    for feature_label in features_to_remove:
        matrix[labeled_features == (feature_label + 1)] = 0  

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

# THERE IS A PROBLEM HERE PROBABLY

def medianBack(matrix, feature_value=0, annulus_width=12):
    
    labeled_features = measure.label(matrix == feature_value)

    medians = []
    for props in measure.regionprops(labeled_features):
        
        min_row, min_col, max_row, max_col = props.bbox

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

# MAGNITUDE DETERMINE IS ALL GOOD
def magDetermine(matrix, median_values):
    
    matrix_copy = np.copy(matrix).astype(float)

    labeled_features = measure.label(matrix != 0)

    for props, median_value in zip(measure.regionprops(labeled_features), median_values):
        
        # Get the coordinates of the feature region
        coords = props.coords
        
        # Subtract the median value only from pixels within the feature region
        for coord in coords:
            matrix_copy[coord[0], coord[1]] -= median_value

    return matrix_copy

def galaxyNum(matrix):
    # Label connected components of features
    labeled_features = measure.label(matrix != 0)

    # Count the number of unique labels (excluding 0, which represents the background)
    num_features = len(np.unique(labeled_features)) - 1

    return num_features

# lets streamline this whole process for some greater clarity in how we
# arrive at the final result, and then quickly put it all into one
# function to simplify the whole process

noBloomValues = no_blooming_bg


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

    adjustedGalaxyMagnitudes = magDetermine(cleanCountGalaxy, countAdjustment)
    
    endItAll = np.copy(adjustedGalaxyMagnitudes)

    countsError = np.sqrt(endItAll)

    mags = fits.PrimaryHDU(endItAll)

    magList = fits.HDUList([mags])

    name = f'{filename}.fits'

    magList.writeto(name, overwrite=True)

    galaxyCount = galaxyNum(endItAll)

    print(galaxyCount)

    return endItAll, galaxyCount, ObjectPositions

def FluxCalibration(inputMatrix):

    with fits.open('mosaic.fits') as hdul:
        # Get the header of the primary HDU (Header/Data Unit)
        header = hdul[0].header

    zeropoint  = header['MAGZPT']
    errorZeroPoint = header['MAGZRR']

    def magEq(counts):
        if counts == 0:
            return 0
        else:
            flux = 3.1 * counts/720
            return zeropoint - 2.5 * np.log10(flux, dtype=float)

    vectorFunc = np.vectorize(magEq)

    magCalibrated = vectorFunc(inputMatrix)

    return magCalibrated

def countSum(image_data):
    labeled_image = measure.label(image_data != 0)

    sum_of_values_per_feature = []

    for prop in measure.regionprops(labeled_image):
        min_row, min_col, max_row, max_col = prop.bbox

        label = prop.label

        region = image_data[min_row:max_row, min_col:max_col]

        sum_of_values = np.sum(region)

        sum_of_values_per_feature.append([label, sum_of_values])

    return sum_of_values_per_feature

def FinalPropagations(inputData):
    sums = countSum(inputData)
    sums = np.array(sums)
    calibratedInfo = FluxCalibration(sums[:, 1])


    freq, bins = np.histogram(calibratedInfo, bins = 150)
    mid = (bins[1:] + bins[:-1]) / 2

    calibratedInfo = sorted(calibratedInfo)

    def CDFFINDER(X):
        return np.count_nonzero(calibratedInfo < X)
    
    vals = np.linspace(min(calibratedInfo), max(calibratedInfo), 400, endpoint = True)
    
    cdf = np.array([CDFFINDER(X) for X in vals])
    logcdf = np.log10(cdf, dtype = float)

    figure, axes = plt.subplots(1, 2, figsize = (20, 15))
    axes[0].plot(vals, logcdf, 'x', color = 'black')
    axes[1].plot(mid, freq, 'o', color = 'blue')

    axes[0].grid(True)
    axes[1].grid(True)

    axes[0].set_title('Cumulative Number Distribution')
    axes[1].set_title('Number Distribution')

    axes[0].set_xlabel('Magnitude')
    axes[0].set_ylabel('log($N(m)$)')

    axes[1].set_xlabel('Magnitude')
    axes[1].set_ylabel('$N(m)$')
    return sums, mid, freq, logcdf


# %%

galaxyMags, totalCount, posGalaxy = SequentialImageProcessor(no_blooming_bg, mu + 3 * std, 50000, mu + 3 * std + 30,  mu, 12, 'FinalFITS')

# %%

mags, binnedMagnitudes, counter, cumulativeSum = FinalPropagations(galaxyMags)

# %%

# def ParameterChecker(stdMultiple, lowerBound, annulus):
#     galaxyMags, totalCount, posGalaxy = SequentialImageProcessor(no_blooming_bg, mu + stdMultiple * std, mu + 3 * std + 1000, mu + 3 * std + lowerBound,  mu, annulus, 'FinalFITS')
#     mags, binnedMagnitudes, counter, cumulativeSum = FinalPropagations(galaxyMags)

#     func, covariance = np.polyfit(binnedMagnitudes, cumulativeSum, 1, cov = True)
#     plotter = np.poly1d(func)

#     return plotter, binnedMagnitudes, cumulativeSum, covariance



# # %%

# fitFunc, xVals, cumsum, cov = ParameterChecker(3, 20, 12)

# %%