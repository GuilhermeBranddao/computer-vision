# Feature Matching

# Imports
import cv2
import numpy as np

# Define os keypoints
def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Cria uma nova imagem de saída que concatene as duas imagens em conjunto
    output_img = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    output_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    output_img[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2, img2])

    # Desenha linhas de conexão entre pontos-chave correspondentes
    for match in matches:
        # Obtém os pontos-chave correspondentes para cada uma das imagens
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Desenha um pequeno círculo em ambas as coordenadas e, em seguida, desenhe uma linha
        radius = 4
        colour = (0,255,0)   # green 
        thickness = 1
        cv2.circle(output_img, (int(x1),int(y1)), radius, colour, thickness)   
        cv2.circle(output_img, (int(x2)+cols1,int(y2)), radius, colour, thickness)
        cv2.line(output_img, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), colour, thickness)

    return output_img

# Função main
if __name__=='__main__':
    img1 = cv2.imread('images/bus-rotated.png', 0)  # query Image
    img2 = cv2.imread('images/bus.png', 0)          # train Image

    #img1 = cv2.imread(sys.argv[1], 0)   # query image 
    #img2 = cv2.imread(sys.argv[2], 0)   # train image 

    # ORB Detector
    orb = cv2.ORB()

    # Extrai keypoints e descritores
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Cria o objeto para fazer o match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Classifiqca na ordem de sua distância
    matches = sorted(matches, key = lambda x:x.distance)

    # Desenha as primeiras 'n' matches
    img3 = draw_matches(img1, keypoints1, img2, keypoints2, matches[:30])

    cv2.imshow('Matched keypoints', img3)
    cv2.waitKey()

