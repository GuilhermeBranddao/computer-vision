# Simple Thresholding

import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
args = vars(ap.parse_args())
 
# Carrega a imagem e converte para Grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplica o filtro gaussiano
# A aplicação do filtro gaussiano ajuda a remover algumas das bordas de alta frequência da imagem 
# que não nos preocupam e nos permitem obter uma segmentação mais "limpa".
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

# Image thresholding com limite 155.
(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

# Vamos aplicar o limiar básico. O primeiro parâmetro é a
# imagem, o segundo valor é o nosso limiar.
# Se um valor de pixel for maior que o nosso limite (neste
# caso, 155), convertemos para BRANCO, caso contrário, PRETO.
(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)

# Finalmente, vamos usar nosso limiar como uma máscara e visualizar apenas as casas na imagem.
cv2.imshow("Casa", cv2.bitwise_and(image, image, mask = threshInv))
cv2.waitKey(0)