# Filtro Customizado
# http://pillow.readthedocs.io/en/4.2.x/reference/ImageFilter.html?highlight=kernel

from PIL import Image
from PIL import ImageFilter

# Leitura da Imagem
img = Image.open("images/beija-flor.jpg")
img.show()

# Filtro
new_img = img.filter(ImageFilter.Kernel((3,3),[1,0,-1,5,0,-5,1,0,1]))
new_img.show()
