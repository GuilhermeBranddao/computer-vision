# Canny Edge Detection

# Imports
from skimage import io
from skimage import feature
from skimage import color

# Imagem
img = io.imread("images/casa.jpg")

# Converte escala de cinza
img = color.rgb2gray(img)

# Canny
edge = feature.canny(img,3)

# Print
io.imshow(edge)
io.show()