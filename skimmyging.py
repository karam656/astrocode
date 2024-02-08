#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:14:02 2024

@author: karam
"""

import numpy as np 
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import skimage as sk
from skimage import measure, morphology
import regionprops
from astrocode import mask
#remember to import stuff from astrocode
#%%
def skimmyging(data, cutoff, area):
    pixelvalues_cropped_01 = mask(data, cutoff)
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
data = pixelvalues_cropped
cutoff = mean_estimate
area = 1900

skimmyging(data, mean_estimate, area)
#%%
# I want to go back to the og data without the hot stuff
def skimmyging_with_change_indicator(data, cutoff, area):
    # Apply the mask to identify bright sources
    pixelvalues_cropped_01 = mask(data, cutoff)
    
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

skim_data, change_indicator = skimmyging_with_change_indicator(pixelvalues_cropped, mean_estimate +3*std, 1850)

#%%
def compare(data, indicator):
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if indicator[i][j]==1:
                data[i][j] = 0
    return data

no_blooming_bg = compare(pixelvalues_cropped, change_indicator)
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
