# Sobel Edge Detection

# Imports
import argparse
import cv2
 
# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
args = vars(ap.parse_args())
 
# Carregando a imagem e convertendo para escala de cinza
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
 
# Computando os gradientes nos eixos X e Y 
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)
 
# As imagens gX e gY são agora do tipo de dados de ponto flutuante,
# então precisamos convertê-las de volta para uma representação não-assinada de 8 bits
# de inteiro para que outras funções OpenCV possam utilizá-las
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)
 
# Combina as representações sobel X e Y em uma única imagem
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
 
# Output
cv2.imshow("Sobel X", gX)
cv2.imshow("Sobel Y", gY)
cv2.imshow("Sobel Combinado", sobelCombined)
cv2.waitKey(0)