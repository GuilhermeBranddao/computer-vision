import cv2

def get_info_image(image):
    print("Altura: {} pixels".format(image.shape[0]))
    print("Largura: {} pixels".format(image.shape[1]))
    print("Canais: {}".format(image.shape[2]))

def show_image(image, winname="Image"):
    cv2.imshow(winname, image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' ou tecla Esc
            break
        
        # também permite fechar com o botão 'X' da janela
        if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def save_image(path_output, image):
    cv2.imwrite(path_output, image)