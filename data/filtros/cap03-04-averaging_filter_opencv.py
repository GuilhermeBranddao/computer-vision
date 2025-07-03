# Averaging Filter

import numpy as np
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carrega a imagem original
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Define uma lista de kernels
kernelSizes = [(3, 3), (9, 9), (15, 15)]
 
# Loop sobre toda a lista de kernels e aplica o Averaging Blur
for (kX, kY) in kernelSizes:
	blurred = cv2.blur(image, (kX, kY))
	cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
	cv2.waitKey(0)