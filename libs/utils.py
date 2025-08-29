
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

def get_pixel(image, coord_xy:tuple):
	"""
	coord_xy = (100,100)
	"""
	return image.getpixel(coord_xy)

def set_pixel_value(img, 
					coord_xy:tuple[int, int], 
					value:tuple[int, int, int], 
					):
	"""
	Teste
	"""
	img.putpixel(coord_xy, value)
	
	return img

def translate(image, x, y):
	"""
	Realiza translação da imagem
	"""
	# Define a matriz de translação e realiza a translação
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Retorna a imagem 
	return shifted

def convert(image, to='grayscale', use="cv2"):
	if use == "skimage":
		if to == "grayscale":
			# Convertendo de RGB para Grayscale
			image = color.rgb2gray(image)
		
		elif to == "hsv":
			image = color.rgb2hsv(image)
	elif use == "pillow":
		# Converte para escala de cinza
		image = image.convert("L")
	

	return image


def crop(image, dim, use="cv2"):
	"""
	dim = (100, 100, 400, 400)
	"""
	if use == "pillow":
		image.crop(dim)
		crop_image = image.crop(dim)
	return crop_image


def rotate(image, angle, center = None, scale = 1.0, use="cv2"):
	"""
	Realiza a rotação da imagem
	"""
	if use == "cv2":
		# Obtém as dimensões da imagem
		(h, w) = image.shape[:2]
		# Se o centro for Nenhum, inicialize-o como o centro da imagem
		if center is None:
			center = (w // 2, h // 2)
		# Executa a rotação
		M = cv2.getRotationMatrix2D(center, angle, scale)
		rotated = cv2.warpAffine(image, M, (w, h))
	elif use == "pillow":
		rotated = image.rotate(angle)

	# Retorna a imagem 
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA, use="cv2"):
	"""
	Realiza a redimencionamento da imagem
	"""
	if use == "cv2":
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
	
	elif use == "pillow":
		resized = image.resize((width, height))

	# Retorna a imagem 
	return resized

def enhance_brightness(image, factor, use="cv2"):
	"""
	Aprimoramento para melhorar o brilho da imagem
	"""
	if use == "pillow":
		# Enhancement
		enhancer = ImageEnhance.Brightness(image)
		bright_image = enhancer.enhance(factor)
	
	return bright_image

def enhance_contrast(image, factor, use="cv2"):
	"""
	Aprimoramento para melhorar o contraste
	"""
	if use == "pillow":
		# Muda o contraste da imagem
		enhancer = ImageEnhance.Contrast(image)
		contrast_image = enhancer.enhance(factor)
	
	return contrast_image


def imshow(winname, image, use="cv2"):
	"""
	Exibe a imagem
	"""
	if use == "cv2":
		cv2.imshow(winname, image)
		while True:
			key = cv2.waitKey(1) & 0xFF
			
			if key == ord('q') or key == 27:  # 'q' ou tecla Esc
				break
			
			# também permite fechar com o botão 'X' da janela
			if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1:
				break

		cv2.destroyAllWindows()

	elif use == "matplotlib":
		plt.imshow(image)
		plt.show()
	
	elif use == "pillow":
		image.show()


def read_image(path, use="cv2"):
	if use == "pillow":
		image = Image.open(path)
	elif use == "cv2":
		image = cv2.imread(path)
	return image

def save_image(path, image, use="cv2"):
	if use == "pillow":
		image.save(path)

	elif use == "cv2":
		cv2.imwrite(path, image)

import numpy as np
from PIL import Image

import cv2
from typing import Union

def apply_grayscale(image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
    """
    Garante que a imagem esteja em escala de cinza e retorna no formato original.

    Parâmetros:
        image: PIL.Image ou NumPy array (OpenCV ou equivalente)

    Retorna:
        Imagem convertida para grayscale no mesmo tipo recebido
    """
    original_type = type(image)

    # --- CASO PIL.Image ---
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            print("Convertendo PIL image para grayscale...")
            image = image.convert('L')
        return image

    # --- CASO NumPy array (cv2 ou outros) ---
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Imagem colorida BGR (OpenCV)
            print("Convertendo NumPy array (possivelmente BGR) para grayscale...")
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image_gray
        elif len(image.shape) == 2:
            # Já está em grayscale
            return image
        else:
            raise ValueError("Formato NumPy de imagem não reconhecido.")
    
    else:
        raise TypeError(f"Tipo de imagem não suportado: {original_type}")

