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
cutoff = mu
area = 1900

skimmyging(data, mu + 3.5 * params[3], area)


# %%