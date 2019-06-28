import os 
import cv2

image=cv2.imread("./train_images/0/0.png")
channel=image.shape[2]
print(channel)
