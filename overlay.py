import cv2

im = cv2.imread('ex.png')
im[0:282, 0:226] = im[218:500, 494:720]

cv2.imshow('im', im)
k = cv2.waitKey(0)
if k == 27:
	exit(0)