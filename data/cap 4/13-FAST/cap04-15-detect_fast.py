# FAST
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html

# Imports
import numpy as np
import cv2

# Carrega e converte para escala de cinza
image = cv2.imread("images/book.jpg")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecta FAST Keypoints
detector = cv2.FastFeatureDetector_create()
kps = detector.detect(gray)
print("Número de keypoints: {}".format(len(kps)))

# Loop pelo keypoints e desenha na imagem
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)


# Print parâmetros
print "Threshold: ", detector.getInt('threshold')
print "nonmaxSuppression: ", detector.getBool('nonmaxSuppression')
print "neighborhood: ", detector.getInt('type')
print "Total Keypoints with nonmaxSuppression: ", len(kps)

# Print
cv2.imshow("Image", np.hstack([orig, image]))
cv2.waitKey(0)