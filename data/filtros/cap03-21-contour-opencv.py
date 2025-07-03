# Countour
# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours

# Imports
import numpy as np
import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Camninho para imagem")
args = vars(ap.parse_args())
 
# Carrega imagem
image = cv2.imread(args["image"])

# Converte para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Desfocagem
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Canny
edged = cv2.Canny(blurred, 30, 150)
 
# Imagem original
cv2.imshow("Original", image)
 
# Encontra todos os contornos na imagem e desenha TODOS os contornos da imagem
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# A função findCountours é destrutiva, ou seja, ela altera a imagem. Por isso fazemos uma cópia.
clone = image.copy()

# Desenha os contornos
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print ("Encontrados {} contornos".format(len(cnts)))
 
# Output
cv2.imshow("Contornos", clone)
cv2.waitKey(0)