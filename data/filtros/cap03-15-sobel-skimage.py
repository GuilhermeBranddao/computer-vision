# Sobel Edge Detection

# Imports
from skimage import io
from skimage import filters
from skimage import color

# Carregando a imagem
img = io.imread("images/casa.jpg")
io.imshow(img)
io.show()

# Convertendo para escala de cinza
img = color.rgb2gray(img)

# Sobel
edge = filters.sobel(img)

# Mostra o resultado
io.imshow(edge)
io.show()
