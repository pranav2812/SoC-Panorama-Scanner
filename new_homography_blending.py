import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# no_of_levels
K=6



def homography(im1,im2):
	
	# print("1")
	

  	# gray1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	orb=cv2.ORB_create(50000)
	kp1,d1=orb.detectAndCompute(im1,None)

	# print("2")

	# gray2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
	kp2,d2=orb.detectAndCompute(im2,None)

	# print("3")

	# bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
	# matches=bf.match(d1,d2)
	# matches=sorted(matches,key=lambda x: x.distance)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(d1,d2,k=2)
	# matches=sorted(matches,key=lambda m: m.distance)

	# print("4")

	good_matches = []
	for m,n in matches:
		if m.distance < n.distance:
			good_matches.append([m])

	# print("5")

	points1=np.zeros((len(good_matches),2),dtype=np.float32)
	points2=np.zeros((len(good_matches),2),dtype=np.float32)

	# print("6")

	# print(points2)

	if len(good_matches) > 4:
		for i,m in enumerate(good_matches):
			points1[i, :]=np.matrix(kp1[m[0].queryIdx].pt)
			points2[i, :]=np.matrix(kp2[m[0].trainIdx].pt)
		# print(points2)

		# print("7")

		h, mask=cv2.findHomography(points1,points2,cv2.RANSAC,5)
	# else None
		# h1,_=cv2.estimateRigidTransform(points1,points2,True)
		# print(h)
		# print(h1)


	# h,mask = cv2.homography_ransac(points1,points2,1000,5)

	# print("8")

	height1, width1, channels1 = im1.shape
	height2, width2, channels2 = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width1+width2, height2),cv2.INTER_NEAREST)
	# im1Reg[0:height2,0:width2] = im2
   
	# merged_img=np.zeros((height1*2,width1,3))

	# print("9")

	# common_region = np.logical_and(im1Reg>0, merged_img>0)
	# print(len(np.where(common_region)[0]))
	# if len(np.where(common_region)[0]) > 0:
	#     avg_warped = np.sum(img_warped[common_region])
	#     avg_merged = np.sum(merged_img[common_region])
	#     pirnt("11111111111")
	#     print(avg_warped, avg_merged)

	#     g_warped = avg_merged/avg_warped
	#     img_warped = np.multiply(img_warped,g_warped)

	# # # Add the warped img to the whole img
	# img_warped[common_region] = 0
	# merged_img = np.add(merged_img,img_warped)
	# warped_imgs.append(img_warped)

	# # Take average of each pixel
	# merged_img = np.divide(merged_img,num_vals)
	# blended_img = multiband_blend(warped_imgs,k)

	# return merged_img,blended_img

	# return im1Reg, h
	im1Reg=cv2.cvtColor(im1Reg,cv2.COLOR_BGR2GRAY)
	im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

	im2Reg=im1Reg[:,0:width2]

	# im1Reg=cv2.resize(im1Reg,None,fx=0.1,fy=0.1,interpolation = cv2.INTER_CUBIC)
	# im2=cv2.resize(im2,None,fx=0.1,fy=0.1,interpolation = cv2.INTER_CUBIC)
	# cv2.imshow('im1Reg',im1Reg)
	# cv2.imshow('im2',im2)
	# cv2.waitKey(0)
	# Gain compensation
	sum_im2Reg=0
	sum_im2=0
	common_region=np.logical_and(im2Reg>0 , im2>0)
	# print(im1Reg)
	if len(np.where(common_region)[0])>0:
		total_final=np.sum(im2[common_region])/np.prod(im2[common_region].shape)
		total_warped=np.sum(im2Reg[common_region])/np.prod(im2Reg[common_region].shape)


		for i in im2Reg[common_region]:
			if total_final-90<i<total_final+90:
				sum_im2Reg=sum_im2Reg+i

		for i in im2[common_region]:
			if total_warped-90<i<total_warped+90:
				sum_im2=sum_im2+i
				
		factor=float(sum_im2Reg)/sum_im2
		# print(im1Reg)
		# factor=1.5
		im1Reg= np.array(im1Reg)/factor
		# print(1/factor)
		# print(im1Reg)
		# plt.imshow("1",im1Reg)
		# plt.show()
		# print(factor)
	# im1Reg[0:height2,0:width2] = im2
	# return im1Reg, h

	im3Reg=im1Reg[:,width2:width2*2]
	
	# generate Gaussian pyramid for A
	G = im2.copy()
	gpA = [G]
	for i in xrange(K):
		G = cv2.pyrDown(G)
		gpA.append(G)

	# generate Gaussian pyramid for im3Reg
	G = im3Reg.copy()
	gpB = [G]
	for i in xrange(K):
		G = cv2.pyrDown(G)
		gpB.append(G)

	# generate Laplacian Pyramid for im2
	lpA = [gpA[K-1]]
	for i in xrange(K-1,0,-1):
		GE = cv2.pyrUp(gpA[i])
		L = cv2.subtract(gpA[i-1],GE)
		lpA.append(L)

	# generate Laplacian Pyramid for im3Reg
	lpB = [gpB[K-1]]
	for i in xrange(K-1,0,-1):
		GE = cv2.pyrUp(gpB[i])
		L = cv2.subtract(gpB[i-1],GE)
		lpB.append(L)



	LS = []
	for la,lb in zip(lpA,lpB):
		# rows,cols,dpt = la.shape
		ls = np.hstack((la[:,0:im2.shape[1]], lb[:,0:im3Reg.shape[1]]))
		LS.append(ls)

	# now reconstruct
	ls_ = LS[0]
	for i in xrange(1,K):
		ls_ = cv2.pyrUp(ls_)
		# LS[i]=cv2.resize(LS[i],None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
		ls_ = cv2.add(ls_, LS[i])
	
	return ls_





if __name__=='__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
	ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
	args = vars(ap.parse_args())

	img1=cv2.imread(args["first"],1)
	img2=cv2.imread(args["second"],1)
	# im1=cv2.resize(img1,None,fx=0.25,fy=0.25,interpolation = cv2.INTER_CUBIC)
	# im2=cv2.resize(img2,None,fx=0.25,fy=0.25,interpolation = cv2.INTER_CUBIC)
	finalimg = homography(img2,img1)
	
	cv2.imwrite("final2.jpg",finalimg)

	# homography(img2,img1)

