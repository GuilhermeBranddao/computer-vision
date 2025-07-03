# Calculando Magnitude e Orientação

# Imports
import numpy as np
import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
ap.add_argument("-l", "--lower-angle", type=float, default=175.0, help="Lower orientation angle")
ap.add_argument("-u", "--upper-angle", type=float, default=180.0, help="Upper orientation angle")
args = vars(ap.parse_args())

# Carrega a imagem e converte para escala de cinza
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
 
# Computando os gradientes nos eixos X e Y 
gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
 
# Calcula a magnitude e a orientação do gradiente, respectivamente
mag = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
 
# Encontra todos os pixels que estão dentro dos limites dos ângulos superior e inferior
idxs = np.where(orientation >= args["lower_angle"], orientation, -1)
idxs = np.where(orientation <= args["upper_angle"], idxs, -1)
mask = np.zeros(gray.shape, dtype="uint8")
mask[idxs > -1] = 255
 
# Output
cv2.imshow("Mask", mask)
cv2.waitKey(0)