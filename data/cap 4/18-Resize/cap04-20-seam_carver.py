# Redimensionamento de Imagens com Base no Conteúdo (Seam Carving)
# http://perso.crans.org/frenoy/matlab2012/seamcarving.pdf
# https://www.youtube.com/watch?v=qadw0BRKeMk
# http://eric-yuan.me/seam-carving/

# Introduzido pela Avidan e Shimar em 2007, o algoritmo Seam Carving é usado para redimensionar uma imagem removendo / adicionando "costuras" (seams) com pouca energia.
# As costuras são definidas como pixels conectados que fluem da esquerda para a direita ou de cima para baixo, desde que atravessem toda a largura / altura da imagem.

# O algoritmo Seam Carving funciona ao encontrar pixels conectados chamados costuras com baixa energia (ou seja, menos importante) que atravessam toda a imagem 
# da esquerda para a direita ou de cima para baixo. Estas costuras são então removidas da imagem original, permitindo-nos redimensionar a imagem, 
# preservando as regiões mais salientes (o algoritmo original também suporta a adição de costuras, o que nos permite aumentar o tamanho da imagem também).

# Tenha em mente que a finalidade do Seam Carving é preservar as regiões mais salientes (ou seja, "interessantes") de uma imagem enquanto ainda redimensiona a imagem em si.
# Métodos tradicionais para redimensionar imagens não se preocupam em determinar as partes interessantes de uma imagem.  O Seam Carving aplica, em vez disso, 
# heurísticas / trajetórias derivadas do mapa de energia para determinar quais regiões da imagem podem ser removidas / duplicadas para garantir que
# (1) todas as regiões "interessantes" da imagem sejam preservadas e 
# (2) isso é feito de forma esteticamente agradável.

# Imports
from skimage import transform
from skimage import filters
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem")
ap.add_argument("-d", "--direction", type=str, default="vertical", help="Direção para remover o seam")
args = vars(ap.parse_args())

# Carrega a imagem e converte para Grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Calcula a representação de magnitude do gradiente Sobel da imagem - isso servirá como nossa entrada do "mapa de energia" para o algoritmo Seam Carving
mag = filters.sobel(gray.astype("float"))
 
# Mostra a imagem original
cv2.imshow("Original", image)

# Loop pelo número de seams para remover
for numSeams in range(20, 180, 20):
    # Executa o Seam Carving, remove o número desejado de frames da imagem - os cortes 'verticais' irão alterar a largura da imagem enquanto os 
    # cortes 'horizontais' mudarão a altura da imagem
    carved = transform.seam_carve(image, mag, args["direction"], numSeams)
    print("Removendo {} seams; novo tamanho: "
        "w={}, h={}".format(numSeams, carved.shape[1],
            carved.shape[0]))
 
    # Output
    cv2.imshow("Carved", carved)
    cv2.waitKey(0)
