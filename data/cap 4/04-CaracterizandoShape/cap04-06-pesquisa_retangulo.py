# Pesquisa Imagem Por Similaridade
# python cap04-06-pesquisa_retangulo.py --dataset output

# Imports
import numpy as np
import cv2
import argparse
import glob
from sklearn.metrics.pairwise import pairwise_distances
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Caminho para o diretório de dataset")
args = vars(ap.parse_args())

# Obtém os caminhos da imagem do disco e inicializa a matriz de dados
imagePaths = sorted(glob.glob(args["dataset"] + "/*.jpg"))
data = []

# Loop por todas as imagens
for imagePath in imagePaths:
	# Carrega a imagem, converte para Grayscale e define um threshold
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

	# Encontra contornos na imagem, mantendo apenas o maior
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key=cv2.contourArea)

	# Extrai o ROI da imagem, redimensiona para um tamanho canônico, computa o vetor de recursos Hu Moments 
	# para o ROI e atualiza a matriz de dados
	(x, y, w, h) = cv2.boundingRect(c)
	roi = cv2.resize(thresh[y:y + h, x:x + w], (50, 50))
	moments = cv2.HuMoments(cv2.moments(roi)).flatten()
	data.append(moments)

# Calcula a distância entre todas as entradas na matriz de dados, em seguida, obtém a soma das distâncias para cada linha, 
# obtendo a linha com a maior distância
# Dado um conjunto de N vetores de recursos, uma função de distância pairwise calculará as distâncias entre cada um 
# dos N vetores de recursos - a saída desta operação é chamada de matriz de distância.
D = pairwise_distances(data).sum(axis=1)
i = np.argmax(D)

# Então, por que a matriz de distância de computação entre os vetores da característica Hu Moment é importante? 
# Bem, uma vez que Momentos de Hu são utilizados para caracterizar a forma de um objeto em uma imagem, podemos assumir 
# que a distância entre os círculos será muito pequena, uma vez que eles são visualmente semelhantes uns aos outros. 
# Por outro lado, podemos assumir que a distância entre o vetor do recurso de retângulo e todos os outros vetores 
# de recursos circulares será muito grande, já que o retângulo não é tão similar quanto todos os outros círculos.

# Print
image = cv2.imread(imagePaths[i])
print ("Encontrei o Retangulo: {}".format(imagePaths[i]))
cv2.imshow("Retangulo", image)
cv2.waitKey(0)

