import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse

#reading image-name from command-line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
 
# load the image
image = cv.imread(args["image"])

#red filter

img_hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
#cv.imshow("a", img_hsv)
# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0
output = cv.cvtColor(output_hsv, cv.COLOR_HSV2BGR)
#######################
#cv.imshow("filtered", output)
#red filtered image --> grayscale
imgray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
#threshold
ret,thresh = cv.threshold(imgray,127,255,cv.THRESH_TOZERO_INV)
#cv.imshow("b&w", thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#Store green contours
contours2 = []
ctr = 0
for i in contours:
	if (len(i)>230):

		peri = cv.arcLength(i, True)
		j = cv.approxPolyDP(i, 0.004 * peri, True)
		
		#####plot green circles
		(x,y),radius = cv.minEnclosingCircle(j)

		center = (int(x),int(y))

		
		contours2.append(center)
		radius = int(radius)
		image = cv.circle(image,center,radius,(0,255,0),2)
		ctr+=1

###########################
print("Number of red arrows :", ctr)
for r in contours2:
	print(r)
###########################NON RED
#original image to grayscale
imgray2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#thresholding
ret2,thresh2 = cv.threshold(imgray2,127,255,0)
#cv.imshow("b$w2", thresh2)
contour4 = []	# red contours
contours3, hierarchy3 = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

#####
for i in contours3:
	if (len(i)>230 and cv.contourArea(i)>500):
		
		contour4.append(i)
		peri = cv.arcLength(i, True)
		j = cv.approxPolyDP(i, 0.004 * peri, True)
		(x,y),radius = cv.minEnclosingCircle(j)
		center = (int(x),int(y))
		radius = int(radius)

		## separate out red ones from these
		flag = True
		if radius<35 or (2*radius>max(image.shape[0], image.shape[1])):
			flag = False
		for k in contours2:
			#if not(center==k):
			if (((center[0]-k[0])**2)+((center[1]-k[1])**2) < 25):
				flag = False
				#print(j)
		if flag:
			#red-circle
			image = cv.circle(image,center,radius,(0,0,255),2)
			print("Position of support aircraft : ", center)
			

cv.imshow("result", image)
cv.waitKey(0)

cv.destroyAllWindows()