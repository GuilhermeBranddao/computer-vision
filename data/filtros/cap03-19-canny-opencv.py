# Canny Edge Detector

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

# Carrega a imagem, converte para escala de cinza e desfoca levemente
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
# Imagem original e desfocada
cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred)
 
# Calcula um limite "wide", "mid-range" e "tight" para as bordas
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)
 
# Mostra os Mapas de Bordas (Map Edges)
cv2.imshow("Wide Edge Map", wide)
cv2.imshow("Mid Edge Map", mid)
cv2.imshow("Tight Edge Map", tight)
cv2.waitKey(0)