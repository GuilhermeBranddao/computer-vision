# Good Features to Track Detector

# O detector de pontos-chave Shi-Tomasi é mais comumente conhecido como Good Features to Track Detector, ou simplesmente o GFTT. 
# Este detector de pontos-chave foi introduzido por Shi e Tomasi em seu artigo de 1994, Good Features to Track (anexo).

# O GFTT é realmente apenas uma modificação muito simples para o detector de pontos-chave Harris. Ele usa uma função 
# de pontuação diferente para melhorar a qualidade geral. Usando este método, podemos encontrar os cantos N mais fortes 
# na imagem dada. Isso é muito útil quando não queremos usar cada canto para extrair informações da imagem.

# Imports
import cv2
import numpy as np

# Leitura da imagem e conversão para Grayscale
img = cv2.imread('images/box.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# GFTT
corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
corners = np.float32(corners)

# Marca os corners
for item in corners:
    x, y = item[0]
    cv2.circle(img, (x,y), 5, 255, -1)

# Print
cv2.imshow("Top 'k' features", img)
cv2.waitKey()
