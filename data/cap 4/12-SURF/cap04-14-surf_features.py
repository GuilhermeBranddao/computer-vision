# SURF
# https://docs.opencv.org/3.3.0/df/dd2/tutorial_py_surf_intro.html

# Imports
import cv2
import numpy as np

# Carrega a imagem e converte para Grayscale
img = cv2.imread('images/plataforma.png')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detector SURF
surf = cv2.xfeatures2d.SURF_create(400)

# Computando os keypoints
kp, des = surf.detectAndCompute(gray, None)
print("Total de keypoints: {}, Shape dos descriptors: {}".format(len(kp), des.shape))

# Desenha os keypoints na imagem
img = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)

# Print
cv2.imshow('SURF Features', img)
cv2.waitKey()
