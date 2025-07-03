# Erosão
# http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.erosion

from skimage import morphology
from skimage import io 

# Leirtura da imagem
img = io.imread('images/beija-flor.jpg')
io.imshow(img)
io.show()

# Aplicando o filtro
eroded_img  = morphology.erosion(img)

# Print
io.imshow(eroded_img)
io.show()
