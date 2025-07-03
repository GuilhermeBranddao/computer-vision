# Median Filter
# Este é um filtro muito simples que retorna o valor médio do pixel e seus vizinhos.

from PIL import Image
from PIL import ImageFilter

# Leitura da imagem
img = Image.open("images/beija-flor.jpg")
img.show()

# Filtro
blur_img = img.filter(ImageFilter.MedianFilter(7))
blur_img.show()

