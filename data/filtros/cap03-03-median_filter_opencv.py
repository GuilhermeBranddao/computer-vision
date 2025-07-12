# Median Filter

import numpy as np
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carrega a imagem
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# A função cv2.medianBlur é usada principalmente para remover o que se chama ruído "sal e pimenta". 
# Ao contrário da Média método mencionado daqui a pouco com o Gaussian Blur, o método da mediana 
# (como o nome sugere), calcula o valor médio do pixel entre os área ao redor.
blurred = np.hstack([
	cv2.medianBlur(image, 3),
	cv2.medianBlur(image, 5),
	cv2.medianBlur(image, 7)])
cv2.imshow("Median", blurred)
cv2.waitKey(0)