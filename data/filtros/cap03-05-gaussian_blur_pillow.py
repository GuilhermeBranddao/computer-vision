# Gaussian Blur
# O Gaussian Blur (ou Desfoque Gaussiano) é um dos filtros mais utilizados no processamento de imagens. 
# Ele usa a curva gaussiana com distribuição em formato de sino.

from PIL import Image
from PIL import ImageFilter

# Leitura da imagem
img = Image.open("images/beija-flor.jpg")
img.show()

# Filtro
blur_img = img.filter(ImageFilter.GaussianBlur(5))
blur_img.show()

# Salvando a imagem
blur_img.save("images/beija-flor-GaussianBlur.jpg")
