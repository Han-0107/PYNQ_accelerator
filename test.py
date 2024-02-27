import bnn

hw_classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW2A2,'distracted_driver',bnn.RUNTIME_HW)
sw_classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW2A2,'distracted_driver',bnn.RUNTIME_SW)

print(hw_classifier.classes)

from PIL import Image
import numpy as np
import cv2

# im = Image.open('./Test_image/deer.jpg')
camera = cv2.VideoCapture(0)
while True:
    # 读取当前帧
    ret,im = camera.read()
    if (not ret):
        break
    class_out=hw_classifier.classify_image(im)
    print("Class number: {0}".format(class_out))
    print("Class name: {0}".format(hw_classifier.class_name(class_out)))

    class_out = sw_classifier.classify_image(im)
    print("Class number: {0}".format(class_out))
    print("Class name: {0}".format(sw_classifier.class_name(class_out)))

# from pynq import Xlnk

# xlnk = Xlnk()
# xlnk.xlnk_reset()
