#!/usr/bin/env python3

import numpy as np
import urllib
import os
import pandas as pd
import cv2

# var for testing
img_spirit = ['referenceImages/'+file for file in os.listdir('referenceImages/') if file.endswith('.png')]
orb = cv2.ORB_create(nfeatures=1500)

def build_image_df(orb, img_spirit):
    """
    input_1: takes in an ORB feature extraction
    input_2: takes in a list of images from your folder
    output: returns a df with keypoints and descriptor
    """
    df = pd.DataFrame() # create new df
    
    # cycle through the images to build the df
    for image in img_spirit:
        # read in the image in the format open cv understands
        #
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)  
        
        # get the keypoints and decriptors of the image
        keypoints, desc = orb.detectAndCompute(img, None)
        
        # add keypoints and desscriptor to a temporary df
        df1 = pd.DataFrame({image: {"keypoints" : keypoints, "descriptor": desc }})
        
        # build original dataframe
        df = pd.concat([df, df1], axis=1)
    return df.T # get transpose of df to get the rows and columns to in the right orientation

def get_distance_sum(match):
    """
    input: takes in the match result of a brute force match
    output: total distance of the first 100 matches
    """
    dist = 0
    match = sorted(match, key=lambda x: x.distance)
    for m in match[:100]:
        dist += m.distance
    return dist

def query(df, image):
    """
    input_1: takes in a dataframe with image names as index and keypoints and descriptors as columns
    input_2: takes in an image to match with images in a df
    output: the name of the image in the df that matches the image
    """
    # read image in a format open cv understands
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    
    # get the keypoints and decriptors of the image
    keypoints, desc = orb.detectAndCompute(img, None)
    
    # create the brute force matcher using Hamming distance and crosscheck true for better results acccourding to OpenCV2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # create a column in the df with the matches
    df[f"matches_{image}"] = df["descriptor"].apply(lambda x: bf.match(desc, x))
    
    # creates a column in the df with the distance
    df[f"closest_{image}"] = df[f"matches_{image}"].apply(get_distance_sum)
    
    # get the record with the smallest distance
    closest = df[df[f"closest_{image}"] == df[f"closest_{image}"].min()]
    return closest.index[0]

spirit_db = build_image_df(orb, img_spirit)
for image in os.listdir("queryImages/"):
    if image.endswith(".jpg") or image.endswith(".png"):
        img = cv2.imread(f"queryImages/{image}", cv2.IMREAD_GRAYSCALE)
        print(image, "------", query(spirit_db, f"queryImages/{image}")) 
