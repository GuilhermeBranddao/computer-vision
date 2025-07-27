
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageEnhance
from skimage import color, io 

def get_info_image(image):
    print("Altura: {} pixels".format(image.shape[0]))
    print("Largura: {} pixels".format(image.shape[1]))
    print("Canais: {}".format(image.shape[2]))

def get_pixel(image, xy:tuple):
	"""
	xy = (100,100)
	"""
	return image.getpixel(xy)

def translate(image, x, y):
	"""
	Realiza translação da imagem
	"""
	# Define a matriz de translação e realiza a translação
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Retorna a imagem 
	return shifted

def convert(image, to='grayscale', open_with="cv2"):
	if open_with == "skimage":
		if to == "grayscale":
			# Convertendo de RGB para Grayscale
			image = color.rgb2gray(image)
		
		elif to == "hsv":
			image = color.rgb2hsv(image)
	elif open_with == "pillow":
		# Converte para escala de cinza
		image = image.convert("L")
	

	return image


def crop(image, dim, open_with="cv2"):
	"""
	dim = (100, 100, 400, 400)
	"""
	if open_with == "pillow":
		image.crop(dim)
		crop_image = image.crop(dim)
	return crop_image


def rotate(image, angle, center = None, scale = 1.0, open_with="cv2"):
	"""
	Realiza a rotação da imagem
	"""
	if open_with == "cv2":
		# Obtém as dimensões da imagem
		(h, w) = image.shape[:2]
		# Se o centro for Nenhum, inicialize-o como o centro da imagem
		if center is None:
			center = (w // 2, h // 2)
		# Executa a rotação
		M = cv2.getRotationMatrix2D(center, angle, scale)
		rotated = cv2.warpAffine(image, M, (w, h))
	elif open_with == "pillow":
		rotated = image.rotate(angle)

	# Retorna a imagem 
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA, open_with="cv2"):
	"""
	Realiza a redimencionamento da imagem
	"""
	if open_with == "cv2":
		# Inicializa as dimensões da imagem a ser redimensionada e obtém o tamanho da imagem
		dim = None
		(h, w) = image.shape[:2]

		# Se tanto a largura quanto a altura são None, então retorna a imagem original
		if width is None and height is None:
			return image

		# Verifica se a largura é Nenhuma
		if width is None:
			# Calcula a proporção da altura e construa as dimensões
			r = height / float(h)
			dim = (int(w * r), height)

		# Caso contrário, a altura é Nenhuma
		else:
			# Calcular a proporção da largura e construir as dimensões
			r = width / float(w)
			dim = (width, int(h * r))

		# Redimensiona a imagem
		resized = cv2.resize(image, dim, interpolation = inter)
	
	elif open_with == "pillow":
		resized = image.resize((width, height))

	# Retorna a imagem 
	return resized

def enhance_brightness(image, factor, open_with="cv2"):
	"""
	Aprimoramento para melhorar o brilho da imagem
	"""
	if open_with == "pillow":
		# Enhancement
		enhancer = ImageEnhance.Brightness(image)
		bright_image = enhancer.enhance(factor)
	
	return bright_image

def enhance_contrast(image, factor, open_with="cv2"):
	"""
	Aprimoramento para melhorar o contraste
	"""
	if open_with == "pillow":
		# Muda o contraste da imagem
		enhancer = ImageEnhance.Contrast(image)
		contrast_image = enhancer.enhance(factor)
	
	return contrast_image


def imshow(winname, image, open_with="cv2"):
	"""
	Exibe a imagem
	"""
	if open_with == "cv2":
		cv2.imshow(winname, image)
		while True:
			key = cv2.waitKey(1) & 0xFF
			
			if key == ord('q') or key == 27:  # 'q' ou tecla Esc
				break
			
			# também permite fechar com o botão 'X' da janela
			if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1:
				break

		cv2.destroyAllWindows()

	elif open_with == "matplotlib":
		plt.imshow(image)
		plt.show()
	
	elif open_with == "pillow":
		image.show()


def read_image(path, open_with="cv2"):
	if open_with == "pillow":
		image = Image.open(path)
	elif open_with == "cv2":
		image = cv2.imread(path)
	return image

def save_image(path, image, open_with="cv2"):
	if open_with == "pillow":
		image.save(path)

	elif open_with == "cv2":
		cv2.imwrite(path, image)