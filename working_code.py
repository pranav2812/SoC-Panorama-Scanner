#Code to stitch 4 images in vertical or horizontal direction
#Usage (in terminal):
# python working_code.py -I=image1dir.extension -II=image2dir.extension -III=image3dir.extension -IV=image4dir.extension -dir=v (or h)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

#Levels of Gaussian pyramid
no_of_levels=2

def get_gaussian(image,no_of_levels):
	layer=image.copy()
	Gauss=[layer]
	for i in range(no_of_levels):
		layer=cv2.pyrDown(layer)
		Gauss.append(layer)
	return Gauss

def get_laplacian(Gauss,no_of_levels):
	layer=Gauss[no_of_levels-1]
	Laplace=[layer]
	for i in xrange(no_of_levels-1,0,-1):
		size=(Gauss[i-1].shape[1],Gauss[i-1].shape[0])
		Gauss_big=cv2.pyrUp(Gauss[i],dstsize=size)
		laplacian=cv2.subtract(Gauss[i-1],Gauss_big)
		Laplace.append(laplacian)
	return Laplace

def join_images(lap1,lap2,no_of_levels):
	joined=[]
	for l1,l2 in zip(lap1,lap2):
		# l2[:l1.shape[0],:l1.shape[1]]=l1
		print(l2.shape)
		for i in range(l2.shape[0]/2):
			for j in range(l2.shape[1]):
				if (l2[i,j]==l2[i,j] and l1[i,j]==l1[i,j]):
					if(l2[i,j]<=4 or l2[i,j]>245):
						l2[i,j]=l1[i,j]
		lap=l2
		joined.append(lap)
	joined_reconstructed=joined[0]
	for i in range(1,no_of_levels):
		size=(joined[i].shape[1],joined[i].shape[0])
		joined_reconstructed=cv2.pyrUp(joined_reconstructed,dstsize=size)
		joined_reconstructed=cv2.add(joined[i],joined_reconstructed)
	return joined_reconstructed

def homography(im1,im2,direction,iterator):

	orb=cv2.ORB_create(30000)
	kp1,d1=orb.detectAndCompute(im1,None)
	kp2,d2=orb.detectAndCompute(im2,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(d1,d2,k=2)
	good_matches = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good_matches.append([m])
	points1=np.zeros((len(good_matches),2),dtype=np.float32)
	points2=np.zeros((len(good_matches),2),dtype=np.float32)
	if len(good_matches) > 4:
		for i,m in enumerate(good_matches):
			points1[i, :]=np.matrix(kp1[m[0].queryIdx].pt)
			points2[i, :]=np.matrix(kp2[m[0].trainIdx].pt)
	h, mask=cv2.findHomography(points1,points2,cv2.RANSAC,5)
	height1, width1 = im1.shape
	height2, width2 = im2.shape
	if direction=='v':
		im1Reg = cv2.warpPerspective(im1, h, (width2, height1+height2),cv2.INTER_NEAREST)
	elif direction=='h':
		im1Reg = cv2.warpPerspective(im1, h, (width2+width1, height2),cv2.INTER_NEAREST)
	else :
		return 0

	print(im1Reg.shape)
	print(im2.shape)
	print(iterator)

	if direction=='h':
		im2_mat=np.zeros((im2.shape[0],im2.shape[1]*iterator/(iterator-1)))
		im2_mat[0:im2.shape[0],0:im2.shape[1]]=im2
	else:
		im2_mat=np.zeros((im2.shape[0]*iterator/(iterator-1),im2.shape[1]))
		im2_mat[0:im2.shape[0],0:im2.shape[1]]=im2

	common_region=np.logical_and(im1Reg>0 , im2_mat>0)
	total_warped = np.sum(im1Reg[common_region])
	total_final = np.sum(im2_mat[common_region])
	factor = float(total_final)/total_warped
	im1Reg=factor*im1Reg

	Gauss2=get_gaussian(im2,no_of_levels)
	Laplace2=get_laplacian(Gauss2,no_of_levels)
	Gauss1=get_gaussian(im1Reg,no_of_levels)
	Laplace1=get_laplacian(Gauss1,no_of_levels)
	final=join_images(Laplace2,Laplace1,no_of_levels)
	return final



if __name__=='__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-I", "--first", required=True,
	help="path to the first image")
	ap.add_argument("-II", "--second", required=True,
	help="path to the second image")
	ap.add_argument("-III", "--third", required=True,
	help="path to the second image")
	ap.add_argument("-IV", "--fourth", required=True,
	help="path to the second image")
	ap.add_argument("-dir", "--direction", required=True,
	help="direction of panorama v(top to bottom) or h(left to right)")
	args = vars(ap.parse_args())
	i=2
	img1=cv2.imread(args["first"],0)
	img2=cv2.imread(args["second"],0)
	img3=cv2.imread(args["third"],0)
	img4=cv2.imread(args["fourth"],0)
	finalimg = homography(img2,img1,args["direction"],i)
	cv2.imwrite("1and2.jpg",finalimg)
	i+=1
	imgi1=cv2.imread("1and2.jpg",0)
	finalimg = homography(img3,imgi1,args["direction"],i)
	cv2.imwrite("1and2and3.jpg",finalimg)
	i+=1
	imgi2=cv2.imread("1and2and3.jpg",0)
	finalimg = homography(img4,imgi2,args["direction"],i)
	
	cv2.imwrite("final2.jpg",finalimg)
