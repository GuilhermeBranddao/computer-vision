# Segmentando Shape de um Objeto em Imagem

# Semelhante aos Momentos de Hu, podemos usar Zernike Moments para caracterizar e quantificar a forma de um objeto em uma imagem. 
# Igualmente parecido com Hu Moments, a forma de uma imagem que desejamos descrever pode ser o contorno (isto é, "limite") 
# da forma ou uma máscara (isto é, "limite preenchido") da forma que queremos descrever. 
# Na maioria das aplicações do mundo real, é comum usar a máscara de forma, pois é menos suscetível ao ruído.

# No entanto, ao contrário de Hu Moments, Zernike Moments são descritores de imagens mais poderosos e geralmente mais precisos, 
# com muito pouco custo computacional adicional. Zernike Moments é um descritor de imagem usado para caracterizar a forma de 
# um objeto em uma imagem. A forma a ser descrita pode ser uma imagem binária segmentada ou o limite do objeto 
# (isto é, o "contorno" da forma).

# Para utilizar e extrair Zernike Momentos, estaremos usando o pacote de mahotas. 
# Zernike Moments não está disponível no OpenCV.

# Execute: python cap04-07-segmentando_shape.py

# Imports
import numpy as np
import cv2
from scipy.spatial import distance as dist
import mahotas

def describe_shapes(image):
	# Inicializa a lista de features de shapes
	shapeFeatures = []

	# Converte para Grayscale e threshold
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (13, 13), 0)
	thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

	# Realiza uma série de dilatações e erosões para fechar furos nas formas
	thresh = cv2.dilate(thresh, None, iterations=4)
	thresh = cv2.erode(thresh, None, iterations=2)

	# Detecta contornos no mapa de borda
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Loop pelos contornos
	for c in cnts:
		# Cria uma máscara vazia para o contorno, e desenha
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		# Extrai o ROI da caixa delimitadora da máscara
		(x, y, w, h) = cv2.boundingRect(c)
		roi = mask[y:y + h, x:x + w]

		# Computa Zernike Moments para o ROI e atualiza a lista de recursos de shape
		features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
		shapeFeatures.append(features)

	# Retorna uma tupla dos contornos e shapes
	return (cnts, shapeFeatures)

# Carrega a imagem de referência contendo o objeto que queremos detectar e descreva a região do shape
refImage = cv2.imread("pokemon_red.png")
(_, gameFeatures) = describe_shapes(refImage)

# Carrega os shapes e descreve cada uma dos objetos na imagem
shapesImage = cv2.imread("shapes.png")
(cnts, shapeFeatures) = describe_shapes(shapesImage)

# Calcula as distâncias euclidianas entre os recursos da imagem do videogame e todas as outras formas na segunda imagem, 
# então encontra o índice da menor distância
D = dist.cdist(gameFeatures, shapeFeatures)
i = np.argmin(D)

# Loop sobre todos os contornos do shape da imagem
for (j, c) in enumerate(cnts):
	# Se o índice do contorno atual não for igual ao contorno de índice do contorno com a menor distância, 
	# desenhe-o na imagem de saída
	if i != j:
		box = cv2.minAreaRect(c)
		box = np.int0(cv2.boxPoints(box))
		cv2.drawContours(shapesImage, [box], -1, (0, 0, 255), 2)

# Desenha a caixa delimitadora em torno da forma detectada
box = cv2.minAreaRect(cnts[i])
box = np.int0(cv2.boxPoints(box))
cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
(x, y, w, h) = cv2.boundingRect(cnts[i])
cv2.putText(shapesImage, "FOUND!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
	(0, 255, 0), 3)

# Output
cv2.imshow("Imagem de Entrada", refImage)
cv2.imshow("Shape Detectado", shapesImage)
cv2.waitKey(0)


