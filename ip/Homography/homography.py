import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def homography(im1,im2):
	gray1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	orb=cv2.ORB_create(1000)
	kp1,d1=orb.detectAndCompute(gray1,None)

	gray2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
	kp2,d2=orb.detectAndCompute(gray2,None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
	matches=bf.match(d1,d2)
	matches=sorted(matches,key=lambda x: x.distance)
	final_matches=[]

	for m in matches:
		if m.distance < m.distance * 0.75:
			final_matches.append((m.trainIdx, m.queryIdx))

	points1=np.zeros((len(matches),2),dtype=np.float32)
	points2=np.zeros((len(matches),2),dtype=np.float32)

	for i,m in enumerate(matches):
		points1[i, :]=kp1[m.queryIdx].pt
		points2[i, :]=kp2[m.trainIdx].pt

	h, mask=cv2.findHomography(points1,points2,cv2.RANSAC)

	height1, width1, channels1 = im1.shape
	height2, width2, channels2 = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width2, height1+height2))
	im1Reg[0:height2,0:width2] = im2
   
	return im1Reg, h

if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
	ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
	args = vars(ap.parse_args())

	img1=cv2.imread(args["first"],1)
	img2=cv2.imread(args["second"],1)
	imReg,h = homography(img2,img1)
	cv2.imwrite("final.jpg",imReg)

