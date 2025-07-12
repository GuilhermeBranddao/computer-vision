# Gerando Imagens Panorâmicas (Pano Stitch)

# Imports
import cv2
import numpy as np
import sys
import argparse

# Argumentos
def argument_parser():
    parser = argparse.ArgumentParser(description='Une duas imagens')
    parser.add_argument("--query-image", dest="query_image", required=True, help="Primeira imagem que precisa ser costurada")
    parser.add_argument("--train-image", dest="train_image", required=True, help="Segunda imagem que precisa ser costurada")
    parser.add_argument("--min-match-count", dest="min_match_count", type=int, required=False, default=10, help="Número mínimo de matches necessários")
    return parser
    
# Warp img2 para img1 usando a matriz de homografia H
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]]) 

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img

if __name__=='__main__':
    #img1 = cv2.imread('images/casa1.png', 0)    # query Image
    #img2 = cv2.imread('images/casa2.png', 0)    # train Image
    args = argument_parser().parse_args()
    img1 = cv2.imread(args.query_image, 0)
    img2 = cv2.imread(args.train_image, 0)
    min_match_count = args.min_match_count

    #cv2.imshow('Query image', img1)
    #cv2.imshow('Train image', img2)
    #cv2.imshow('Frame 1', img1)
    #cv2.imshow('Frame 2', img2)

    # SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Extrai keypoints e descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Inicializa os parâmetros para o Flann based matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # Inicializa o objeto Matcher baseado em Flann
    # O matcher baseado em Flann é mais rápido do que a combinação Brute Force porque não compara cada ponto com cada ponto na outra lista. 
    # Só considera a vizinhança do ponto atual para obter o ponto-chave correspondente, tornando-o mais eficiente. 
    # Uma vez que obtemos uma lista de pontos-chave correspondentes, usamos o teste de proporção da Lowe para manter apenas as correspondências fortes. 
    # David Lowe propôs este teste de razão para aumentar a robustez do SIFT.
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Computa os matches
    # Basicamente, quando combinamos os pontos-chave, rejeitamos as correspondências em que a proporção das distâncias para o vizinho mais próximo e 
    # o segundo vizinho mais próximo é maior que um determinado limite. Isso nos ajuda a descartar os pontos que não são distintos o suficiente. 
    # Então, usamos esse conceito aqui para manter apenas as boas correspondências e descartar o resto. Se não tivermos correspondências suficientes, não avançaremos. 
    # No nosso caso, o valor padrão é 10. Você pode brincar com este parâmetro de entrada para ver como isso afeta a saída.
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Armazena todas as boas combinações de acordo com o teste de proporção Lowe
    good_matches = []
    for m1,m2 in matches:
        if m1.distance < 0.7*m2.distance:
            good_matches.append(m1)

    if len(good_matches) > min_match_count:
        src_pts = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result = warpImages(img2, img1, M)
        cv2.imshow('Output', result)

        cv2.waitKey()

    else:
        print ("Não temos número suficiente de correspondências (matches) entre as duas imagens.")
        print ("Encontrado apenas %d correspondências. Precisamos de pelo menos %d correspondências." % (len(good_matches), min_match_count))

