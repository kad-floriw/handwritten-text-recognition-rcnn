import cv2
import random
import numpy as np


def preprocess(img, imgSize, dataAugmentation=False):
	# increase dataset size by applying random stretches to the images
	if dataAugmentation:
		stretch = (random.random() - 0.5)  # -0.5 .. +0.5
		wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
		img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

	# create target image and copy sample image into it
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)

	# scale according to f (result at least 1 and at most wt or ht)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
	img = cv2.resize(img, newSize)
	target = np.ones([ht, wt]) * 255
	target[:newSize[1], :newSize[0]] = img

	# transpose for TF
	img = cv2.transpose(target)

	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s > 0 else img

	return img
