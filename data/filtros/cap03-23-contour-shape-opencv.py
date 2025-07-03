# Countour

# Imports
import numpy as np
import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Canminho para imagem")
args = vars(ap.parse_args())
 
# Carrega imagem
image = cv2.imread(args["image"])

# Converte para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Encontra os contornos externos
(_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Loop por todos os contornos
for c in cnts:
	# Aproxima o contorno
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
 
	# Se o contorno aproximado tiver 4 vértices, então estamos examinando um retângulo
	if len(approx) == 4:
		# Desenha o contorno e desenha o texto na imagem
		cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
		(x, y, w, h) = cv2.boundingRect(approx)
		cv2.putText(image, "Retangulo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
 
# Output
cv2.imshow("Imagem", image)
cv2.waitKey(0)