# Adaptive Thresholding

import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
args = vars(ap.parse_args())
 
# Carrega a imagem e converte para Grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Filtro gaussiano
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

# Neste exemplo vamos examinar um vizinho do pixel e de forma adaptativa
# aplicar um limite para cada vizinho. Neste exemplo, iremos
# calcular o valor médio da área de vizinhança de 11 pixels
# e limitar com base nesse valor. Finalmente, nosso C constante é
# subtraído do cálculo médio (neste caso 4)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Threshold", thresh)

# We can also apply Gaussian thresholding in the same manner
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Threshold", thresh)
cv2.waitKey(0)