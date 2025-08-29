import numpy as np
from PIL import Image
from libs.utils import apply_grayscale
import cv2
from typing import Union

def equelize_histogram(image:Image.Image) -> Image.Image:
    """
    Equaliza o histograma de uma imagem em escala de cinza manualmente.

    Etapas:
    1. Calcular o histograma da imagem (256 bins).
    2. Calcular a função de distribuição cumulativa (CDF).
    3. Normalizar a CDF para o intervalo [0, 255].
    4. Usar a CDF como função de mapeamento para os pixels.

    Parâmetros:
        image: PIL.Image em escala de cinza ('L').

    Retorna:
        Imagem equalizada (PIL.Image)
    """

    image = apply_grayscale(image)
    

    image_array = np.array(image)
    flat = image_array.flatten()

    # 1. Histograma
    hist, _ = np.histogram(flat, bins=256, range=[0, 256])

    # 2. CDF (Função de Distribuição Acumulada)
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, value=0) # Evita divisão por zero

    # 3. Normaliza a CDF
    cdf_normalizer = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalizer, 0).astype(np.uint8)

    # 4. Mapear os valores antigos para os novos usado a CDF
    equalized = cdf_final[image_array]

    return Image.fromarray(equalized)


def clahe(image: Image.Image, tile_size: int = 64, clip_limit: float = 0.01) -> Image.Image:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) em uma imagem em escala de cinza.

    CLAHE divede a imagem em pequenos blocos (tiles), equeliza localmente e combina os blocos suavemente
    
    Parâmetros:
        image: PIL.Image em escala de cinza ('L')
        tile_size: tamanho dos blocos (ex: 64x64)
        clip_limit: valor entre 0 e 1. Fração máxima de pixels por bin do histograma

    Retorna:
        PIL.Image com equalização adaptativa
    """

    image = apply_grayscale(image)
    image_array = np.array(image)
    height, width = image_array.shape
    image_zeros_like = np.zeros_like(image_array)
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):

            # Recorta tile
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            image_tile = image_array[y:y_end, x:x_end]


            flat = image_tile.flatten()
            hist, _ = np.histogram(flat, bins=256, range=[0, 256])

            # --- C: Limita o histograma ---
            n_pixels = image_tile.size
            limit = int(clip_limit * n_pixels)

            # Clip nos bins
            excess = np.sum(np.maximum(hist - limit, 0))
            hist = np.minimum(hist, limit)

            # Redistribui excesso uniformemente
            hist += excess // 256

            # CDF com histograma ajustado
            cdf = hist.cumsum()
            cdf_masked = np.ma.masked_equal(cdf, 0)
            cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
            cdf_final = np.ma.filled(cdf_normalized, 0).astype(np.uint8)

            image_zeros_like[y:y_end, x:x_end] = cdf_final[image_tile]
    
    return Image.fromarray(image_zeros_like)

def clahe_interpolated(image: Image.Image, tile_size: int = 64, clip_limit: float = 0.01) -> Image.Image:
    """
    CLAHE com interpolação bilinear para reduzir o efeito de mosaico (blocos visíveis).

    Parâmetros:
        image: PIL.Image em grayscale
        tile_size: tamanho dos blocos (ex: 64)
        clip_limit: valor de corte para limitar o histograma local

    Retorna:
        Imagem equalizada com interpolação entre tiles
    """
    image = apply_grayscale(image)
    img_array = np.array(image)
    h, w = img_array.shape

    n_tiles_y = (h + tile_size - 1) // tile_size
    n_tiles_x = (w + tile_size - 1) // tile_size

    # Calcula LUTs para cada tile
    luts = []
    for i in range(n_tiles_y):
        row = []
        for j in range(n_tiles_x):
            y_start = i * tile_size
            x_start = j * tile_size
            y_end = min((i + 1) * tile_size, h)
            x_end = min((j + 1) * tile_size, w)

            tile = img_array[y_start:y_end, x_start:x_end]
            flat = tile.flatten()
            hist, _ = np.histogram(flat, bins=256, range=[0, 256])

            # Limitação do histograma (clip_limit)
            limit = int(clip_limit * flat.size)
            excess = np.sum(np.maximum(hist - limit, 0))
            hist = np.minimum(hist, limit)
            hist += excess // 256

            cdf = hist.cumsum()
            cdf = np.ma.masked_equal(cdf, 0)
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf = np.ma.filled(cdf, 0).astype(np.uint8)
            row.append(cdf)
        luts.append(row)

    # Aplica interpolação para cada pixel
    result = np.zeros_like(img_array)

    for y in range(h):
        for x in range(w):
            # Índices dos tiles ao redor
            ty = y // tile_size
            tx = x // tile_size
            dy = (y % tile_size) / tile_size
            dx = (x % tile_size) / tile_size

            def get_lut(i, j):
                i = min(max(i, 0), n_tiles_y - 1)
                j = min(max(j, 0), n_tiles_x - 1)
                return luts[i][j]

            # Luts dos 4 tiles ao redor
            lut00 = get_lut(ty, tx)
            lut10 = get_lut(ty, tx + 1)
            lut01 = get_lut(ty + 1, tx)
            lut11 = get_lut(ty + 1, tx + 1)

            val = img_array[y, x]

            # Interpola os 4 vizinhos
            top = (1 - dx) * lut00[val] + dx * lut10[val]
            bot = (1 - dx) * lut01[val] + dx * lut11[val]
            interp = (1 - dy) * top + dy * bot

            result[y, x] = int(interp)

    return Image.fromarray(result)

def clahe_color(image: Image.Image, tile_size: int = 64, clip_limit: float = 0.01) -> Image.Image:
    """
    Aplica CLAHE no canal Y (luminância) de uma imagem RGB.

    Parâmetros:
        image: PIL.Image colorida
        tile_size: tamanho do bloco (tile)
        clip_limit: limite de contraste

    Retorna:
        Imagem PIL com CLAHE aplicado no brilho
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    y_channel = img_yuv[:, :, 0]
    y_pil = Image.fromarray(y_channel)

    y_eq = clahe_interpolated(y_pil, tile_size=tile_size, clip_limit=clip_limit)
    img_yuv[:, :, 0] = np.array(y_eq)

    img_bgr_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    img_rgb_eq = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_rgb_eq)
