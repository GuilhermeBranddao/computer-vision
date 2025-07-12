# Harris Corner Detector

# Este algoritmo específico foi desenvolvido para identificar o canto interno de uma imagem. 
# Os cantos são regiões dentro de uma imagem em que há grandes variações na intensidade do gradiente em todas as direções.

# O Harris Corner Detector foi introduzido em 1988 por Harris e Stephens em seu documento, 
# A Combined Corner and Edge Detector (anexo). Este detector é um dos detectores de canto mais comuns que você 
# encontrará no mundo da visão computacional. É bastante rápido (não tão rápido como o detector de ponto-chave FAST), 
# mas é mais preciso ao marcar as regiões como cantos.

# Imports
import cv2
import numpy as np

# Leitura da image e conversão para Grayscale
img = cv2.imread('images/box.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Conversão para float32
gray = np.float32(gray)

# Harris Corner Detector
# cv2.cornerHarris(image, blockSize, ksize, k)

# Esta função recebe quatro argumentos.

# img - imagem a ser analisada, deve estar em escala de cinza e com valores float32.
# blockSize - tamanho das janelas consideradas para a detecção de canto
# ksize - parâmetro para a derivada de Sobel
# k - parâmetro livre para a equação de Harris.

corners = cv2.cornerHarris(gray, 4, 5, 0.04)       # Corners
#corners = cv2.cornerHarris(gray,9,5,0.04)      # Corner inferior direito
#corners = cv2.cornerHarris(gray,14,5,0.04)     # Corner superior esquerdo

# O resultado é dilatado para marcar os cantos
corners = cv2.dilate(corners,None)

# Limiar para um valor ideal; pode variar dependendo da imagem.
img[corners>0.01*corners.max()]=[0,0,0]

cv2.imshow('Harris Corners',img)
cv2.waitKey()


