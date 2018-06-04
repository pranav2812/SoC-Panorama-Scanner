import cv2
import matplotlib.pyplot as plt

# max_features1=10
# max_features2=10
image1='23.jpg'
image2='25.jpg'
no_of_maps=1000

def make_connections(i1,i2):
	
	#make images grayscale from bgr
	gray1=cv2.cvtColor(i1,cv2.COLOR_BGR2GRAY)
	gray2=cv2.cvtColor(i2,cv2.COLOR_BGR2GRAY)
	
	#mark orbs
	orb1=cv2.ORB_create()
	kp1,d1=orb1.detectAndCompute(gray1,None)
	orb2=cv2.ORB_create()
	kp2,d2=orb2.detectAndCompute(gray2,None)

	#match orbs
	# matcher=cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	# matches=matcher.match(d2,d1,None)
	bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

	#sorting
	# matches.sort(key=lambda x: x.distance)
	matches=bf.match(d1,d2)
	matches=sorted(matches,key=lambda x: x.distance)

	#taking good factor matches
	# end=int(len(matches)*factor_of_matches)
	matches=matches[:no_of_maps]

	#drawing
	imMatches=cv2.drawMatches(i1,kp1,i2,kp2,matches,None)
	cv2.imwrite("matched.jpg",imMatches)
	# plt.imshow(imMatches)
	# plt.show()
	final_image=cv2.imread('matched.jpg',1)
	cv2.imshow('final_image',final_image)
	cv2.waitKey(0)

if __name__=='__main__':
	i1=cv2.imread(image1,1)
	i2=cv2.imread(image2,1)
	make_connections(i1,i2)
