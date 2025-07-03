# Dilatação

import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
args = vars(ap.parse_args())
 
# Carrega a imagem e converte para Grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
 
# Aplica a erosão
for i in range(0, 3):
	dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
	cv2.imshow("Dilated {} times".format(i + 1), dilated)
	cv2.waitKey(0)