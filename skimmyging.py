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