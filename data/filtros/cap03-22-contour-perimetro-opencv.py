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
clone = image.copy()
 
# Loop por todos os contornos
for (i, c) in enumerate(cnts):
	# Calcula a área e o perímetro do contorno
	area = cv2.contourArea(c)
	perimeter = cv2.arcLength(c, True)
	print ("Contorno #%d -- Area: %.2f, Perimetero: %.2f" % (i + 1, area, perimeter))
 
	# Desenha o contorno na imagem
	cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
 
	# Calcula o centro do contorno e desenha o número do contorno
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	cv2.putText(clone, "#%d" % (i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (255, 255, 255), 4)
 
# Output
cv2.imshow("Contours", clone)
cv2.waitKey(0)