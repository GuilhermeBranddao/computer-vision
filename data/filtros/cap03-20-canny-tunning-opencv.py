# Canny Edge Detector Tunning

def auto_canny(image, sigma=0.33):
	# Calcula a mediana das intensidades de pixel de um único canal 
	v = np.median(image)
 
	# Aplicando a detecção automática de borda Canny usando a mediana calculada
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# Retorna a borda da imagem
	return edged

# Imports
import numpy as np
import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
args = vars(ap.parse_args())

# Carrega a imagem, converte para escala de cinza e desfoca levemente
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
# Aplica a detecção de borda Canny usando um limiar wide, tight e automático
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)
 
# Print 
cv2.imshow("Original", image)
cv2.imshow("Wide", wide)
cv2.imshow("Tight", tight)
cv2.imshow("Auto", auto)
cv2.waitKey(0)