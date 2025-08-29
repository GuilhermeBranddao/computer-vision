# Retrieving Images and Searching Using Image Descriptors

Descritores são features principais extraidas por determinados algoritmos 
Essas features podem ser usados em dases de dados afim de procurar key points para stitch images

This chapter will show you to detect the features of an image with OpenCV and make use of them to match and search images

More specifically, we will cover the following tasks:
- Detecting keypoints and extracting local descriptors around the keypoints usingm any of the following algorithms: Harris corners, SIFT, SURF, or ORB
- Matching keypoints using brute-force algorithms or the FLANN algorithm
- Filtering out bad matches using KNN and the ratio test
- Finding the homography between two sets of matching keypoints
- Searching a set of images to determine which one contains the best match for a reference image

## Understanding types of feature detection and matching

- **Harris**: This algorithm is useful for detecting corners.
- **SIFT**: This algorithm is useful for detecting blobs.
- **SURF**: This algorithm is useful for detecting blobs.
- **FAST**: This algorithm is useful for detecting corners.
- **BRIEF**: This algorithm is useful for detecting blobs.
- **ORB**: This algorithm stands for Oriented FAST and Rotated BRIEF. It is useful for detecting a combination of corners and blobs.

Matching features can be performed with the following methods:
- Brute-force matching
- FLANN-based matching



