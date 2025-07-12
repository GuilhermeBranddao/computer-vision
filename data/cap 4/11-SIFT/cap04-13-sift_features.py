# SIFT
# https://docs.opencv.org/3.3.0/da/df5/tutorial_py_sift_intro.html

# Imports
import cv2
import numpy as np

# Carregando a imagem e convertendo para Grayscale
input_image = cv2.imread('images/plataforma.png')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Criando o detector de recursos SIFT
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray_image, None)

# Desenhando os keypoints na imagem
input_image = cv2.drawKeypoints(input_image, keypoints, gray_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostra a forma dos pontos-chave e a matriz de descritores invariantes locais
kp, des = sift.detectAndCompute(gray_image,None)
print("Número de keypoints Detectados: {}".format(len(kp)))
print("Shape do Vetor de Recursos: {}".format(des.shape))

# Print
cv2.imshow('SIFT Features', input_image)
cv2.waitKey()

