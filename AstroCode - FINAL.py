#%%

# Astropy, skimage (sci-kit) may need to be pip-installed
import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, poisson
from scipy.optimize import curve_fit

import skimage as sk
from skimage import measure
from scipy.stats import ks_2samp

#%%

#loading in data
hdulist = fits.open("mosaic.fits")

pixelvalues2 = hdulist[0].data

pixelvalues_cropped_horizontally = pixelvalues2[:, 120:-120]

# Crop vertically by slicing only rows from 110 to -110, keeping all columns
# this to get rid of the image borders and noisy edge data in the image itself.
pixelvalues = pixelvalues_cropped_horizontally[110:-110, :]

#%%
#plotting histogram, mask used to remove anomolies/ clean data
# narrows data down to just our background/feature space, 
# removes super bright, or super dim data

mask = (3000 < pixelvalues) & (pixelvalues < 4000)

# Use the mask to select pixels within the desired range
selected_pixels = pixelvalues[mask]


# histogram is clearly asymmetric, as expected from a poisson distribution
# best to use a skewed gaussian, as fitting a poisson was technically difficult
def skewed_gaussian(x, amp, a, mean, std):
    return amp * skewnorm.pdf(x, a, loc=mean, scale=std)

# extract key values
hist, bins = np.histogram(selected_pixels, bins = 1000, density=True)

midpoints = (bins[1:] - bins[:-1])/2

# fit values
params, covariance = curve_fit(skewed_gaussian, bins[:-1], hist, p0 = (900000, 1.4, 3425, np.sqrt(3425)), maxfev = 100000)

# plot to show results. Mean = global background value
plt.figure(figsize=(10, 10))
plt.bar(bins[:-1], hist, width = np.diff(bins), edgecolor='black', label = 'Data ')
plt.plot(bins[:-1], skewed_gaussian(bins[:-1], *params), label = 'Gauss fit', color = 'red')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.legend()
plt.grid()
plt.xlim(3300,3700)
plt.show()



#%%

# produce binary dataset that gets rid of any pixels that are below our threshold
# threshold was chosen to be mean + 3 * std

# this allows us to do further processing

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
cutoffSet = mu + 3 * std
Apass = 1800


#%%


# I want to go back to the data without the ultra-bright/blooming/bleeding

# this uses some scripts to return the local backgrounds as appropriate while 
# removing ultra-bright star, and any other blooming/bleeding

# remove objects that have an area greater than or equal to 
# 1850px^2

def skimmyging_with_change_indicator(data, cutoff, area):

    # Create a copy of the data to work with
    # prevents python messing up memory and storages

    storage = np.copy(data)
    
    # Apply the mask to identify bright sources
    pixelvalues_cropped_01 = masker(storage, cutoff)
    
    # Label the objects in the masked image
    label_image, num_features = measure.label(pixelvalues_cropped_01, background=0, return_num=True)
    
    # Initialize the change indicator array with zeros
    change_indicator = np.zeros(storage.shape, dtype=int)
    
    # skimage does a lot of the heavy lifting for is
    regions = measure.regionprops(label_image)

    for region in regions:
        if region.area >= area:
            for coordinates in region.coords:
                storage[coordinates[0], coordinates[1]] = 0
                change_indicator[coordinates[0], coordinates[1]] = 1

    # to see exactly what has been removed.
    
    # concurrent with what is desired
                
    plt.imshow(change_indicator, cmap = 'gray')
    plt.show()

    return storage, change_indicator

skim_data, changer = skimmyging_with_change_indicator(pixelvalues, cutoffSet, 1850)

#%%

# by using the aforementioend, where the coordinates of the large, bright
# objects have been found, its indices are used here to remove them by
# setting their values to zero in the actual data.

# as such, background and other noise relevant later is preserved.

def compare(data, indicator):
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if indicator[i][j]==1:
                data[i][j] = 0
    return data

no_blooming_bg = compare(pixelvalues, changer)
#plotting this isn't too nice

#%%
#trying to save to a fits file 

# if you see this in DS9, it is very clear that the job is done well

hdu = fits.PrimaryHDU(no_blooming_bg)
hdulist2 = fits.HDUList([hdu])
filename = 'no_blooming_bg.fits'
hdulist2.writeto(filename, overwrite=True)

# %%

# PHOTOMETRY SECTION BEGINS HERE


# all the "extractor" modules are there to appropriately extract information
# while preserving the shape of the matrix to avoid any confusions later

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
    # getting wrongly updated, so have to do it this way, didn't want to risk
    # changing to np.copy and that

    matOut = np.zeros_like(matrix)
    matOut += matrix
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    # to set all the hot-pixels to the mean background radiation value 
    # of 3409 AKA mu - this only helps in our galaxy analysis

    # later, we will see how using a process without this is necessary
    # for error analysis

    matOut[row_indices, col_indices] = setValue
    
    return matOut


# this function is important for finding the indices of the hot pixels
# these indices are then used later for further processing

# hot pixels are usually isolated, and so have no adjacent non-zero pixels
# and so the connectivity feature on sk-image is used here to do the heavy-lifting
def lonelyRemove(binary_array):

    labeled_array, num_features = measure.label(binary_array, connectivity=2, return_num=True)

    # Count sizes of each connected component
    sizes = np.bincount(labeled_array.ravel())

    # Get indices of components with size 1 (lone elements)
    lone_element_indices = np.where(sizes <= 5)[0]

    for idx in lone_element_indices:
        labeled_array[labeled_array == idx] = 0

    filtered_binary_array = labeled_array > 0

    non_lone_positions = np.argwhere(filtered_binary_array)

    return filtered_binary_array, non_lone_positions


# exists simply to establish thresholds on counts
# data is unreliable above 50000 counts due to detector limitations
# so any features (connectivity) with a maximum pixel value > 50000 is removed

# likewise, some noise from earlier blooming removal is still present
# putting a lower threshold to remove these here

# cannot simply change mu + 3 * std as the local backgrounds are all different
# making it impossible to do correctly back then

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

# combines lonelyRemove and FinalClearance

# returns galaxy indices
def combinedRemove(binaryArray, matrix, upperBound, lowerBound):
    # Apply FinalClearance on the matrix

    cleaned_matrix, posIn = lonelyRemove(binaryArray)

    inputVal = extractorNonBool(matrix, posIn)

    final_matrix, non_lone_positions = FinalClearance(inputVal, threshold=upperBound, threshold2=lowerBound)

    # Remove lonely elements from the cleaned matrix

    return cleaned_matrix, final_matrix, non_lone_positions


# uses a rectangular annulus around any features (same connectivity 
# method to identify) take median to prevent any clipping issues 

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

# Median values are index matches to indices of the features.
# easy to go through, use same library to subtract median values from 
# corresponding features/galaxies
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

# count number of galaxies found after all
def galaxyNum(matrix):
    # Label connected components of features
    labeled_features = measure.label(matrix != 0)

    # Count the number of unique labels 
    # (excluding 0, which represents the background)
    num_features = len(np.unique(labeled_features)) - 1

    return num_features

# lets streamline this whole process for some greater clarity in how we
# arrive at the final result, and then quickly put it all into one
# function to simplify the whole process

noBloomValues = no_blooming_bg

# combines all the above
def SequentialImageProcessor(data, backgroundBounds, starBounds, dustBounds, 
                             meanBackground, annulusWidth, filename):
    
    # find indices of pixels above mean + 3 * std value 
    # - essentially just the features
    galaxyIndex = data > backgroundBounds
    booleanGalaxy = np.array(galaxyIndex).astype(int)

    # find indices of pixels below mean + 3 * std value
    # - essentially just hte background
    backgroundIndex = data <= backgroundBounds
    booleanBackground = np.array(backgroundIndex).astype(int)

    # actually find the data of these indices in counts, instead of just binary
    galaxies = extractor(data, galaxyIndex)
    background = extractor(data, backgroundIndex)

    # remove hotpixels etc, and remove upper and lower threshold data as required
    cleanBinaryGalaxies, cleanCountGalaxy, ObjectPositions = \
    combinedRemove(booleanGalaxy, galaxies, starBounds, dustBounds)
    
    # invert cleaned out (no-hot-pixel) data from the binary feature-only dataset
    # find where this non-hotpixel and our aformentioned booleanbackground 
    # values mismatch

    inverted = np.logical_not(cleanBinaryGalaxies)
    hotPixels = np.argwhere((inverted & (booleanBackground == 0)))

    # use these to set hot pixels to background, producing a uniform background only
    # dataset
    cleanBackground = extractorBackground(background, hotPixels, meanBackground)

    # features/galaxies in cleanBackground are zero-valued.
    # find these using skimage, and use a rectangular annulus to calculate median
    # background around these galaxies
    countAdjustment = medianBack(cleanBackground, feature_value=0, annulus_width=annulusWidth)

    # remove local background from the galaxy's counts to correctly
    # find galaxy's magnitude in terms of 
    adjustedGalaxyMagnitudes = magDetermine(cleanCountGalaxy, countAdjustment)
    
    endItAll = np.copy(adjustedGalaxyMagnitudes)

    # save to see in DS9 clearly
    mags = fits.PrimaryHDU(endItAll)
    magList = fits.HDUList([mags])
    name = f'{filename}.fits'
    magList.writeto(name, overwrite=True)

    # print number of observed galaxies
    galaxyCount = galaxyNum(endItAll)

    print(galaxyCount)

    return endItAll, galaxyCount, ObjectPositions

# using equation ins cript and header data to calcualte flux appropriately
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

# need sum of allc ounts per feature to correctly calculate
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

# important fit functions
def schechterFinal(magnitude, phi_star, m_star, alpha):
    return phi_star * np.exp(-10**(0.4 * (magnitude - m_star))) * (10**(0.4 * (magnitude - m_star)))**(alpha+1)

def ideal(N, C):
    ideal = 0.6*np.array(N) + C
    return ideal

# statistical tests, plotting, curve-fitting and many more
def FinalPropagations(inputData):

    # ------------------------------------------------------------
    # Actual Data Procesing and Value Propagations
    sums = countSum(inputData)
    sums = np.array(sums)
    calibratedInfo = FluxCalibration(sums[:, 1])

    # ------------------------------------------------------------
    # extracting relevant information to see the actual number distributions
    freq, bins = np.histogram(calibratedInfo, bins = 50)
    freqNormalised, bins = np.histogram(calibratedInfo, bins = 50, density = True)
    
    mid = (bins[1:] + bins[:-1]) / 2
    bin_width = np.diff(bins)[0]
    calibratedInfo = sorted(calibratedInfo)

    # ------------------------------------------------------------

    def CDFFINDER(X):
        return np.count_nonzero(calibratedInfo < X)
    
    vals = np.linspace(min(calibratedInfo), max(calibratedInfo), 30, endpoint = True)
    
    cdf = np.array([CDFFINDER(X) for X in vals])

    # discrete, smaller intervalled cumulative sum propagation
    logcdf = np.ma.log10(cdf, dtype = float)

    #csum used to have same dimensions as calibrated info for easy calculations
    csum = np.ma.log10(np.linspace(1, len(calibratedInfo), len(calibratedInfo)))


    # ------------------------------------------------------------
    # Error Propagation

    fluxpercerr = np.sqrt(vals)/vals
    x_errperc = np.sqrt(0.08 ** 2 + fluxpercerr ** 2)
    
    #y error of the second plot

    y_err1 = np.sqrt(freq) / freq  # Y-axis error calculation
    y_err1[freq == 0] = 0
    
    x_err1 = bin_width/4
    
    # ------------------------------------------------------------
    # Calculating values of all our fits/linear etc, and also Doing the Curve_fitting
    
    # Linear Fitting - Ideal Equation from LabScript
    p0f, cov = curve_fit(ideal, calibratedInfo[50:800], csum[50:800], p0 = (-17), maxfev = 100000)
    labIdeal = ideal(calibratedInfo, p0f)

    # Generally Fitting a Line to Our CDF - I dont know the actual physical significance
    coeffs, residualsPoly, _, _, _ = np.polyfit(calibratedInfo[50:800], csum[50:800], 1, full = True)
    fitlin = np.poly1d(coeffs)
    ydat = fitlin(calibratedInfo)


    sXdata = mid
    sYdata = freqNormalised

    schecParams, schecCov = curve_fit(schechterFinal, sXdata, sYdata, [0.001, 23.2, -1.01], maxfev = 100000)

    def AmpScaler(magnitude, amp):
        return amp * schechterFinal(magnitude, *schecParams)

    finalSchec, finalCov = curve_fit(AmpScaler, mid, freq, p0 = 320, maxfev = 100000)

    # ------------------------------------------------------------
    # This is all the chi2 statistical testing

    residuals = ((csum[50:800] - labIdeal[50:800])/np.sqrt(csum[50:800]))
    chi2lab = np.sum(residuals**2)
    dof = len(calibratedInfo[50:800]) - len(p0f)
    reducedChi2lab = chi2lab/dof

    chi2linear = np.sum((residualsPoly/np.sqrt(csum))**2)
    dof2 = len(calibratedInfo) - len(coeffs)
    reducedChi2linear = chi2linear/dof2

    # ------------------------------------------------------------
    # This is all the KS-2 Testing

    DlabValue, pValueLab = ks_2samp(csum, labIdeal)
    print(f'Lab Equation - Fixed Gradient - KS-2 Test D : {DlabValue}, p : {pValueLab}')

    DlinValue, pValueLin = ks_2samp(csum, ydat)
    print(f'Linear Fit - Variable Gradient - KS-2 Test D : {DlinValue}, p : {pValueLin}')

    # DschecValue, pSchecValue = ks_2samp(csum, SchechterFunction(calibratedInfo, *pGeneralSchec))
    # print(f'Schecter Function KS-2 Test D : {DschecValue}, p : {pSchecValue}')

    # ----------------------------------------------------------
    # This is for all the plotting 

    figure, axes = plt.subplots(1, 2, figsize = (20, 15))

    axes[0].plot(calibratedInfo, ydat, alpha = 1, color = 'red', label='Linear Fit')
    axes[0].plot(calibratedInfo, ideal(calibratedInfo, p0f), '-', alpha = 0.5, color = 'grey', label='Ideal Equation (2)')    
    axes[0].errorbar(vals, logcdf, yerr = logcdf*np.sqrt(cdf)/cdf, xerr=x_errperc , fmt = 'x', color = 'black', capsize = 5, label = 'Data')
    
    axes[1].errorbar(mid, freq, xerr = x_err1, yerr = y_err1*freq, fmt = 'o', color = 'blue', capsize = 5, label='Data')
    axes[1].plot(mid, finalSchec * schechterFinal(mid, *schecParams), label = 'Schecter Fit')
    
    axes[0].grid(True)
    axes[1].grid(True)

    axes[0].set_title('Cumulative Number Distribution')
    axes[1].set_title('Number Distribution')

    axes[0].set_xlabel('Magnitude')
    axes[0].set_ylabel('log($N(m)$)')

    axes[1].set_xlabel('Magnitude')
    axes[1].set_ylabel('$N(m)$')

    axes[0].legend()
    axes[1].legend()
    return sums, mid, freq, logcdf, vals, reducedChi2lab, reducedChi2linear

# %%

galaxyMags, totalCount, posGalaxy = SequentialImageProcessor(no_blooming_bg, mu + 3 * std, 50000, mu + 3 * std + 30,  mu, 12, 'FinalFITS')

# %%

plt.figure(figsize = (10, 10), dpi = 900)
plt.imshow(galaxyMags[1100:1300, 1000:1200])
plt.xlabel('Pixel Coordinates (X)', fontsize = 15)
plt.ylabel('Pixel Coordinates (Y)', fontsize = 15)
plt.title('Processed Image Snapshot', fontsize = 15)

# %%

mags, binnedMagnitudes, counter, cumulativeSum, vals, chi2lab, chi2lin = FinalPropagations(galaxyMags)


# %%
def SchechterPropagations(inputData):
    # ------------------------------------------------------------
    # Actual Data Procesing and Value Propagations
    sums = countSum(inputData)
    sums = np.array(sums)
    calibratedInfo = FluxCalibration(sums[:, 1])

    # ------------------------------------------------------------
    # extracting relevant information to see the actual number distributions
    freq, bins = np.histogram(calibratedInfo, bins = 40)
    freqNormalised, bins = np.histogram(calibratedInfo, bins = 40, density = True)
    
    mid = (bins[1:] + bins[:-1]) / 2
    bin_width = np.diff(bins)[0]
    calibratedInfo = sorted(calibratedInfo)

    # ------------------------------------------------------------

    def CDFFINDER(X):
        return np.count_nonzero(calibratedInfo < X)
    
    vals = np.linspace(min(calibratedInfo), max(calibratedInfo), 30, endpoint = True)
    
    cdf = np.array([CDFFINDER(X) for X in vals])

    # discrete, smaller intervalled cumulative sum propagation
    logcdf = np.ma.log10(cdf, dtype = float)

    #csum used to have same dimensions as calibrated info for easy calculations
    csum = np.ma.log10(np.linspace(1, len(calibratedInfo), len(calibratedInfo)))

    
    # ------------------------------------------------------------
    # Error Propagation

    fluxpercerr = np.sqrt(vals)/vals
    x_errperc = np.sqrt(0.08 ** 2 + fluxpercerr ** 2)
    
    #y error of the second plot

    y_err1 = np.sqrt(freq) / freq  # Y-axis error calculation
    y_err1[freq == 0] = 0
    
    x_err1 = bin_width/2
    
    # ------------------------------------------------------------
    # Calculating values of all our fits/linear etc, and also Doing the Curve_fitting
    
    # Linear Fitting - Ideal Equation from LabScript
    p0f, cov = curve_fit(ideal, calibratedInfo[50:800], csum[50:800], p0 = (-17), maxfev = 100000)
    labIdeal = ideal(calibratedInfo, p0f)

    # Generally Fitting a Line to Our CDF - I dont know the actual physical significance
    coeffs, residualsPoly, _, _, _ = np.polyfit(calibratedInfo[50:800], csum[50:800], 1, full = True)
    fitlin = np.poly1d(coeffs)
    ydat = fitlin(calibratedInfo)


    sXdata = mid
    sYdata = freqNormalised

    schecParams, schecCov = curve_fit(schechterFinal, sXdata, sYdata, [0.001, 23.2, -1.01], maxfev = 100000)

    def AmpScaler(magnitude, amp):
        return amp * schechterFinal(magnitude, *schecParams)

    finalSchec, finalCov = curve_fit(AmpScaler, mid, freq, p0 = 320, maxfev = 100000)

    # ------------------------------------------------------------
    # Schechter Function's CDF Calculation:
    
    width = np.diff(vals)[0]

    PDFscaled = schechterFinal(vals, *schecParams)

    CDFcheck = np.ma.log10(np.cumsum(PDFscaled * width)) + logcdf[-1]


    # ------------------------------------------------------------
    # This is all the chi2 statistical testing

    residuals = ((csum[50:800] - labIdeal[50:800])/np.sqrt(csum[50:800]))
    chi2lab = np.sum(residuals**2)
    dof = len(calibratedInfo[50:800]) - len(p0f)
    reducedChi2lab = chi2lab/dof

    chi2linear = np.sum((residualsPoly/np.sqrt(csum))**2)
    dof2 = len(calibratedInfo) - len(coeffs)
    reducedChi2linear = chi2linear/dof2


    # ------------------------------------------------------------
    # This is all the KS-2 Testing

    DlabValue, pValueLab = ks_2samp(csum, labIdeal)
    print(f'Lab Equation - Fixed Gradient - KS-2 Test D : {DlabValue}, p : {pValueLab}')

    DlinValue, pValueLin = ks_2samp(csum, ydat)
    print(f'Linear Fit - Variable Gradient - KS-2 Test D : {DlinValue}, p : {pValueLin}')

    DschecValue, pSchecValue = ks_2samp(logcdf, CDFcheck)
    print(f'Schecter Function KS-2 Test D : {DschecValue}, p : {pSchecValue}')

    # ----------------------------------------------------------
    # This is for all the plotting 

    linearStart = 16.4
    linearEnd = 23.8

    plt.figure(figsize=(15, 10))

    #plt.plot(calibratedInfo, ydat, alpha=1, color='red', label='Linear Fit')
    plt.plot(calibratedInfo, ideal(calibratedInfo, p0f), '-', alpha=0.5, color='orange', label='non-Evolution Model')    
    plt.errorbar(vals, logcdf, yerr=logcdf*np.sqrt(cdf)/cdf, xerr=x_err1, fmt='x', color='black', capsize=3, label='Data')
    plt.plot(vals, CDFcheck, color = 'blue', linestyle = 'dashed', label='Schechter Fit - CDF')
    plt.grid(True)
    plt.axvline(linearStart, linestyle = 'dotted', color = 'black', alpha = 0.4)
    plt.axvline(linearEnd, linestyle = 'dotted', color = 'black', alpha = 0.4)
    
    why = np.linspace(0, 4.5, 20)
    plt.fill_betweenx(why, x1 = 15.2, x2 = linearStart, color = 'blue', alpha = 0.15, label = 'Sample-Limited')
    plt.fill_betweenx(why, x1 = linearEnd, x2 = 26, color = 'green', alpha = 0.15, label = 'CCD-Limited')
    plt.fill_betweenx(why, x1 = linearStart, x2 = linearEnd, color = 'gray', alpha = 0.15
                      , label = 'non-evolution Regime')

    plt.title('Cumulative Number Distribution', fontsize = 15)
    plt.xlabel('Magnitude', fontsize = 15, loc = 'right')
    plt.ylabel('log($N(m)$)', fontsize = 15, loc = 'top')
    plt.legend(fontsize = 15, loc = 'upper left')
    plt.ylim(0, 4.5)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlim(15.2, 26)

    ax_small = plt.axes([0.575, 0.175, 0.30, 0.30])  # Create a smaller set of axes at the specified position within the main plot area
    ax_small.errorbar(mid, freq, yerr=y_err1*freq, xerr=x_err1, fmt='o', color='blue', label='Frequency')
    ax_small.plot(mid, finalSchec * schechterFinal(mid, *schecParams), color = 'red', label = 'Schechter PDF')
    ax_small.set_title('Galaxy Number Distribution', fontsize = 15)
    ax_small.set_xlabel('Magnitude', fontsize=15, loc = 'right')
    ax_small.set_ylabel('Frequency', fontsize=15, loc = 'top')
    ax_small.legend(fontsize = 15)
    ax_small.grid()
    plt.show()

    return schecParams, schecCov, calibratedInfo

# %%

pS, cS, cInfo = SchechterPropagations(galaxyMags)


# %%

# COMPLETENESS ANALYSIS

# USED TO FIND THE AMPLITUDE FACTOR WHEN CALCULATING ACTUAL NUMBER FROM
# SCHECHTER FUNCTION
def homogenous(x):
    return 0.6 * (x - 16)


a = np.linspace(10, 21, 50, endpoint = True)
wA = np.diff(a)[0]
theoCDF = homogenous(a)


plt.plot(a, theoCDF)
num = np.cumsum(10**theoCDF * wA)
num[-1]

# %%

magi = np.linspace(min(binnedMagnitudes), 22.5, 50, endpoint = True)

w = np.diff(magi)[0]

PDFs = schechterFinal(magi, 1, 19.5, -1.1)

CDFc = np.ma.log10(np.cumsum(PDFs * w)) + np.log10(num[-1])

print(10 ** CDFc[-1])

plt.plot(magi, CDFc)
plt.xlim((16.0, 22.5))


# %%