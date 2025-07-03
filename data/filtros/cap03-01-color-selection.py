# Detectando Contornos na imagem

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Leitura da imagem e print de algumas estatísticas
image = mpimg.imread('images/estrada.jpg')
print('Esta imagem é um objeto: ',type(image), 'com as seguintes dimensões:', image.shape)
plt.imshow(image)
plt.show()

# Obtém o tamanho de x e y e faz uma cópia da imagem
xsize = image.shape[0]
ysize = image.shape[1]
print('Linhas: ', xsize)
print('Colunas: ', ysize)

# Dica: sempre faça uma cópia da imagem antes de qualquer alteração na imagem original
color_select = np.copy(image)

# Define os critérios de seleção de cores (o valor 230 foi escolhido aleatoriamente)
red_threshold = 230
green_threshold = 230
blue_threshold = 230

# Limites das cores RGB
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identifica os pixels abaixo do limite (threshold)
# Depois disso, todos os pixels que atendam ao nosso critério de cores (aqueles acima do limite) serão mantidos 
# e aqueles que não atendem (abaixo do limite) serão apagados.
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

# O resultado, color_select, é uma imagem na qual os pixels que estavam acima do limite foram retidos 
# e os pixels abaixo do limite foram apagados.
color_select[thresholds] = [0,0,0]

# Print da imagem             
plt.imshow(color_select)
plt.show()