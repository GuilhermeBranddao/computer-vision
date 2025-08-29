from typing import Union
from scipy.ndimage import uniform_filter
from PIL import Image
import numpy as np

def threshold_global(image: Image.Image, T: int = 127, max_val: int = 255, return_type: str = "PIL") -> Union[Image.Image, np.ndarray]:
    """
    Aplica thresholding global a uma imagem em escala de cinza.

    Parâmetros:
        image (PIL.Image): Imagem em escala de cinza ('L').
        T (int): Valor do limiar (threshold).
        max_val (int): Valor máximo atribuído aos pixels acima do limiar.
        return_type (str): 'PIL' para retornar uma imagem PIL.Image ou 'array' para retornar um NumPy array.

    Retorna:
        Imagem binária em preto e branco, como PIL.Image ou np.ndarray.
    """
    if image.mode != 'L':
        raise ValueError("A imagem deve estar no modo 'L' (grayscale).")

    img_array = np.array(image)
    binary_array = np.where(img_array > T, max_val, 0).astype(np.uint8)

    if return_type == "array":
        return binary_array
    return Image.fromarray(binary_array)


def threshold_adaptive_mean(image: Image.Image, block_size: int = 11, C: int = 5) -> Image.Image:
    """
    Thresholding adaptativo: usa a média local da vizinhança como limiar para cada pixel.

    Ideal para imagens com iluminação não uniforme.
    
    Parâmetros:
        image: PIL.Image em escala de cinza ('L')
        block_size: tamanho da janela local (ímpar)
        C: valor subtraído da média local

    Retorna:
        Imagem binária como PIL.Image
    """
    if image.mode != 'L':
        raise ValueError("A imagem deve estar em modo 'L' (grayscale).")
    if block_size % 2 == 0:
        raise ValueError("block_size deve ser ímpar.")

    img_array = np.array(image).astype(np.float32)

    # Média local usando filtro uniforme
    mean = uniform_filter(img_array, size=block_size)
    
    # Aplica thresholding adaptativo
    binary = np.where(img_array > (mean - C), 255, 0).astype(np.uint8)
    
    return Image.fromarray(binary)

def threshold_otsu_manual(image: Image.Image) -> Image.Image:
    """
    Threshold global usando o método de Otsu (implementação manual), usando histograma e variância entre classes.
    
    Parâmetros:
        image: PIL.Image em escala de cinza ('L')

    Retorna:
        Imagem binária como PIL.Image
    """
    if image.mode != 'L':
        raise ValueError("A imagem deve estar em modo 'L' (grayscale).")
    
    img_array = np.array(image)
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    total = img_array.size

    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    sum_b, weight_b = 0.0, 0.0

    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue

        weight_f = total - weight_b
        if weight_f == 0:
            break

        sum_b += t * hist[t]

        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f

        # Variância entre classes
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2

        if var_between > current_max:
            current_max = var_between
            threshold = t

    binary = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def threshold_percentile(image: Image.Image, percentile: float = 85.0) -> Image.Image:
    """
    Aplica thresholding baseado em um percentil da imagem.
    
    Parâmetros:
        image: PIL.Image em escala de cinza ('L')
        percentile: valor percentual (ex: 85.0 aplica threshold no valor que separa os 85% mais escuros)

    Retorna:
        Imagem binária como PIL.Image
    """
    if image.mode != 'L':
        raise ValueError("A imagem deve estar em modo 'L' (grayscale).")

    img_array = np.array(image)
    T = np.percentile(img_array, percentile)
    binary = np.where(img_array > T, 255, 0).astype(np.uint8)

    return Image.fromarray(binary)

