# Estatísticas dos Canais de Cores
# Para executar essse script, use: cap04-02-color_channel_stats.py

# Imports
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import paths

# Obtém a lista de imagem e inicializa o índice para armazenar o nome do arquivo da imagem e o vetor de características
imagePaths = sorted(list(paths.list_images("elefantes")))
index = {}

# Loop por todas as imagens
for imagePath in imagePaths:
	# Carrega a imagem e extrai o nome do arquivo
	image = cv2.imread(imagePath)
	filename = imagePath[imagePath.rfind("/") + 1:]

	# Extraia a média e desvio padrão de cada canal da imagem BGR, depois atualiza o índice com o vetor de características
	(means, stds) = cv2.meanStdDev(image)
	features = np.concatenate([means, stds]).flatten()
	index[filename] = features

# Exibe a imagem da consulta e obtém as chaves ordenadas do dicionário de índice
query = cv2.imread(imagePaths[0])
cv2.imshow("Query (elefante01.jpg)", query)
keys = sorted(index.keys())

# Loop por todos os arquivos do dicionário
for (i, k) in enumerate(keys):
	# Se for a imagem query, ignora
	if k == "elefante01.jpg":
		continue

	# Carrega a imagem atual e calcula a distância euclidiana entre a imagem da consulta (ou seja, a 1ª imagem) e a imagem atual
	image = cv2.imread(imagePaths[i])
	d = dist.euclidean(index["elefante01.jpg"], index[k])

	# Exibe a distância entre a imagem da consulta e a imagem atual
	cv2.putText(image, "%.2f" % (d), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow(k, image)

# Aguarda pressionar um tecla para encerrar
cv2.waitKey(0)



