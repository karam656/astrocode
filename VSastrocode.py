#%%
import numpy as np 
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
#%%
#loading in data
hdulist = fits.open("mosaic.fits")
pixelvalues = hdulist[0].data
#%%
#plottinh histogram, mask used to remove anomolies/ clean data
mask = (3000 < pixelvalues) & (pixelvalues < 4000)

# Use the mask to select pixels within the desired range
selected_pixels = pixelvalues[mask]

hist, bins = np.histogram(selected_pixels, bins = 1000)

plt.figure(figsize=(8, 6))
plt.bar(bins[:-1], hist, width = np.diff(bins), edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.xlim(3300,3700)
plt.show()

std_noise = np.std(selected_pixels)

bin_midpoints = (bins[:-1] + bins[1:]) / 2

# Calculate the weighted sum of midpoints
weighted_sum = np.sum(hist * bin_midpoints)

# Total number of data points
total_counts = np.sum(hist)

# Calculate the mean
mean_estimate = weighted_sum / total_counts
#the mean of this histogram should be like the background noise
#%%
#create a mask array, same size as the image where 1 will indicate images
#that are dealt with and 0 will be the unsearched regions

def mask(data, cutoff):
    mask_arr = np.zeros(data.shape)
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            
            if data[i][j] >= cutoff:
                mask_arr[i][j] = 1
    return mask_arr

testdata = np.array([[0,0,1,2,5,1.1,0],[2,1,3,0,0,0.9],[2,0,1]])

test1 = mask(testdata, 1)
#%%
def mask(data, cutoff):
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
test1 = mask(testdata, 1)
#%%
