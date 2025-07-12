# Gera imagens
# Execute: python cap04-05-generate-images.py --output output

# Imports
import numpy as np
import cv2
import argparse
import uuid

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Caminho para o diretório")
ap.add_argument("-n", "--num-images", type=int, default=500, help="Número de imagens geradas")
args = vars(ap.parse_args())

# Loop pelo número de imagens
for i in range(0, args["num_images"]):
	# Aloca a memória para a imagem e, em seguida, gere o centro (x, y) do círculo e, em seguida, gere o raio do círculo, 
	# garantindo que o círculo esteja totalmente contido na imagem
	image = np.zeros((500, 500, 3), dtype="uint8")
	(x, y) = np.random.uniform(low=105, high=405, size=(2,)).astype("int0")
	r = np.random.uniform(low=25, high=100, size=(1,)).astype("int0")[0]

	# Gerar uma cor para o círculo, desenha e grava a imagem no arquivo usando um nome de arquivo aleatório
	color = np.random.uniform(low=0, high=255, size=(3,)).astype("int0")
	print(color)
	cv2.circle(image, (x, y), r, (0,255,0), -1)
	cv2.imwrite("{}/{}.jpg".format(args["output"], uuid.uuid4()), image)

# Aloca memória para a imagem do retângulo e, em seguida, gere as coordenadas inicial e final (x, y) 
image = np.zeros((500, 500, 3), dtype="uint8")
topLeft = np.random.uniform(low=25, high=225, size=(2,)).astype("int0")
botRight = np.random.uniform(low=250, high=400, size=(2,)).astype("int0")

# Desenha o retângulo na imagem e grava no arquivo usando um nome de arquivo aleatório
color = np.random.uniform(low=0, high=255, size=(3,)).astype("int0")
cv2.rectangle(image, tuple(topLeft), tuple(botRight), (0,255,0), -1)
cv2.imwrite("{}/{}.jpg".format(args["output"], uuid.uuid4()), image)