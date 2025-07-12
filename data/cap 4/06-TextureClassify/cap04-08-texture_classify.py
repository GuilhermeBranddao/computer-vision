# Classificação de Textura com Haralick features

# Os Haralick features são usados para descrever a textura de uma imagem. 
# A textura refere-se à aparência, consistência ou "sensação" de uma superfície. 
# Exemplos de texturas incluem "áspero" versus "macio". 
# As aplicações potenciais das Haralick features incluem determinar se uma estrada é pavimentada versus cascalho.

# As Haralick features são calculadas usando a Matriz de Co-ocorrência de Nível de Cinza (GLCM). 
# Esta matriz caracteriza a textura, registrando a frequência com que os pares de pixels adjacentes com valores 
# específicos ocorrem em uma imagem.

# Execute: python cap04-08-texture_classify.py --training training --test testing

# Imports
from sklearn.svm import LinearSVC
import argparse
import mahotas
import glob
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Caminho para o dataset de texturas")
ap.add_argument("-t", "--test", required=True, help="Caminho para as imagens de teste")
args = vars(ap.parse_args())

# Inicializa a matriz de dados e a lista de labels
print ("Extraindo os recursos...")
data = []
labels = []

# Loop sobre o dataset de imagens de treino
for imagePath in glob.glob(args["training"] + "/*.png"):
	# Carrega a imagem, converte em escala de cinza e extrai o nome da textura do nome do arquivo
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	texture = imagePath[imagePath.rfind("/") + 1:].split("_")[0]

	# Extrai as características da textura de Haralick em 4 direções, depois calcula a média de cada direção
	features = mahotas.features.haralick(image).mean(axis=0)

	# Atualiza dados e labels
	data.append(features)
	labels.append(texture)

# Treinando o Classificador
print ("Treinando o modelo...")
model = LinearSVC(C=10.0, random_state=42)
model.fit(data, labels)
print ("Classificando...")

# Loop pelas imagens de teste
for imagePath in glob.glob(args["test"] + "/*.png"):
	# Carrega a imagem, converte em escala de cinza e extrai a textura Haralick da imagem de teste
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	features = mahotas.features.haralick(gray).mean(axis=0)

	# Classifica a imagem de teste
	pred = model.predict(features.reshape(1, -1))[0]
	cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

	# Output
	cv2.imshow("Image", image)
	cv2.waitKey(0)

