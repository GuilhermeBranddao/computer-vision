# Caracterizando a Forma de um Objeto em uma Imagem com Momentos Estatísticos
# Extraindo HuMoments

# Do ponto de vista estritamente estatístico, "momentos" são apenas expectativas estatísticas de uma variável aleatória. 
# Na verdade, você provavelmente já está familiarizado com pelo menos um momento, tenha você percebido ou não!

# O momento mais comum é o primeiro momento, a média. Você também está muito familiarizado com o segundo momento, 
# a variância - e tomar a raiz quadrada da variância nos deixa com o desvio padrão, que provavelmente você também 
# está familiarizado. Inclinação (skewness) e curtose (kurtosis) completam os terceiro e quarto momentos, respectivamente.

# O descritor Hu Moments retorna um vetor de características de valor real de sete valores (momentos). 
# Esses sete valores capturam e quantificam a forma do objeto em uma imagem. 
# Podemos então comparar o shape do nosso vetor de características com outros vetores de recursos para 
# determinar como duas formas são "semelhantes".

# Execute: python cap04-04-extract-humoments.py

# Imports
import cv2

# Carrega e converte para Grayscale
image = cv2.imread("images/planes.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calcula o vetor de recursos Hu Moments para toda a imagem (NÃO É O IDEAL)
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print ("ORIGINAL MOMENTS: {}".format(moments))
cv2.imshow("Image", image)
cv2.waitKey(0)

# Encontra os contornos dos três planos na imagem
(_, cnts, _) = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop por cada contorno
for (i, c) in enumerate(cnts):
	# Extrai o ROI da imagem e computa o vetor de recursos de Hu Moments para ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = image[y:y + h, x:x + w]

	# Calcula recursos Hu Moments para as imagens delimitadas pelos contornos (ISSO É O IDEAL)
	moments = cv2.HuMoments(cv2.moments(roi)).flatten()

	# Print moments e ROI
	print ("MOMENTS FOR PLANE #{}: {}".format(i + 1, moments))
	cv2.imshow("ROI #{}".format(i + 1), roi)
	cv2.waitKey(0)


