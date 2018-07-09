#Code to stitch 2 images in vertical or horizontal direction
#Usage (in terminal):
# python working_code.py -I=image1dir.extension -II=image2dir.extension -dir=v (or h)



import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

#Levels of Gaussian pyramid
no_of_levels=3


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

def join_images(lap1,lap2,no_of_levels,mask):
	joined=[]
	goto=0
	for l1,l2 in zip(lap1,lap2):
		
		size=(l2.shape[1],l2.shape[0])
		if goto!=0:
			mask=cv2.pyrUp(mask,dstsize=size)
		inv_mask=np.bitwise_not(mask)
		print("Naruto")
		lap=cv2.add(cv2.bitwise_and(l1,mask),cv2.bitwise_and(l2,inv_mask))
		joined.append(lap)
		if goto==0:
			goto=1
	joined_reconstructed=joined[0]
	for i in range(1,no_of_levels):
		size=(joined[i].shape[1],joined[i].shape[0])
		joined_reconstructed=cv2.pyrUp(joined_reconstructed,dstsize=size)
		joined_reconstructed=cv2.add(joined[i],joined_reconstructed)
	print("Arigatou")
	return joined_reconstructed

def homography(im1,im2,direction,iterator):

	orb=cv2.ORB_create(50000)
	kp1,d1=orb.detectAndCompute(im1,None)
	kp2,d2=orb.detectAndCompute(im2,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(d1,d2,k=2)
	good_matches = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
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

	print(iterator)

	if direction=='h':
		im2_mat=np.zeros((im2.shape[0],im2.shape[1]*iterator/(iterator-1)),dtype=np.uint8)
		im2_mat[0:im2.shape[0],0:im2.shape[1]]=im2
	else:
		im2_mat=np.zeros((im2.shape[0]*iterator/(iterator-1),im2.shape[1]),dtype=np.uint8)
		im2_mat[0:im2.shape[0],0:im2.shape[1]]=im2

	common_region=np.logical_and(im1Reg>0 , im2_mat>0)
	true_section1=np.where(common_region==True)
	print(true_section1)
	# factor = float(total_final)/total_warped
	# factor=np.array([factor],dtype=np.float64)
	# print(factor)
	# im1Reg=np.multiply(im1Reg,factor)
	# im1Reg=np.array(im1Reg,dtype=np.uint8)

	print("Yos")

	Gauss2=get_gaussian(im2_mat,no_of_levels)
	Laplace2=get_laplacian(Gauss2,no_of_levels)
	Gauss1=get_gaussian(im1Reg,no_of_levels)
	Laplace1=get_laplacian(Gauss1,no_of_levels)
	mask=np.full((im2_mat.shape[0],im2_mat.shape[1]),255,dtype=np.uint8)
	print("Nani!?")

	mask=cv2.bitwise_and(mask,im2_mat)
	mask[np.nonzero(mask)]=255
	print("Dattebayo!")
	Gauss_mask=get_gaussian(mask,no_of_levels)
	Gauss_mask=np.flip(Gauss_mask,0)
	print("Hai")
	final=join_images(Laplace2,Laplace1,no_of_levels,Gauss_mask[1])
	return final



if __name__=='__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-I", "--first", required=True,
	help="path to the first image")
	ap.add_argument("-II", "--second", required=True,
	# help="path to the second image")
	# ap.add_argument("-III", "--third", required=True,
	# help="path to the second image")
	# ap.add_argument("-IV", "--fourth", required=True,
	help="path to the second image")
	ap.add_argument("-dir", "--direction", required=True,
	help="direction of panorama v(top to bottom) or h(left to right)")
	args = vars(ap.parse_args())
	i=2
	img1=cv2.imread(args["first"],0)
	img2=cv2.imread(args["second"],0)
	# img3=cv2.imread(args["third"],0)
	# img4=cv2.imread(args["fourth"],0)
	finalimg = homography(img2,img1,args["direction"],i)
	# cv2.imwrite("1and2.jpg",finalimg)
	i+=1
	# imgi1=cv2.imread("1and2.jpg",0)
	# finalimg = homography(img3,imgi1,args["direction"],i)
	# cv2.imwrite("1and2and3.jpg",finalimg)
	# i+=1
	# imgi2=cv2.imread("1and2and3.jpg",0)
	# finalimg = homography(img4,imgi2,args["direction"],i)
	
	cv2.imwrite("final2.jpg",finalimg)
