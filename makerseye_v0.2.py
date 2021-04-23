#!/usr/bin/python3

#Version 0.2
#Changes: Cropping sleeves, pull from jnet, languages, box of interest
#Should detect cards on a blank background in all orientations
#added invert mode

import cv2
import numpy as np
import pickle
from PIL import Image
import imagehash
import urllib.request

#languages are en, zh-simp
LANG = 'en'
STDEVS = 5
CROP_PX = 0

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		if(refPt[0][0] == refPt[1][0] or refPt[0][1] == refPt[1][1]):
			refPt = []
			return 0
		a = (max(refPt[0][0],refPt[1][0]), max(refPt[0][1],refPt[1][1]))
		b = (min(refPt[0][0],refPt[1][0]), min(refPt[0][1],refPt[1][1]))
		print(a,b)
		refPt = [b, a]
		
		cropping = False

def hamming(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

#LOAD HASHES
f = open("scans.32ihash","rb")
d = pickle.load(f)
f.close()

def identify(img,raw):
	#PROCESSING
#	thresh = cv2.equalizeHist(img)
#	thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
#	cv2.imshow("can",thresh)
#	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#	contours = sorted(contours, key=lambda x: -cv2.arcLength(x,True))
#	hull = cv2.convexHull(contours[0])
	blur = cv2.medianBlur(img, 5)
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 11, 7)

	kernel = np.ones((3,3))
	dilate = cv2.erode(thresh, kernel, iterations=1)
#	dilate = cv2.dilate(dilate, kernel, iterations=1)

	cv2.imshow("can",dilate)

	contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if(len(contours) > 0):
		hull = cv2.convexHull(np.concatenate(contours))
	else:
		return 0

	#FIT RECTANGLE
	epsilon = 0.1*cv2.arcLength(hull,True)
	quad = cv2.approxPolyDP(hull,epsilon,True)
	#cv2.drawContours(img, cont, -1, (0,0,0), 1)
	cv2.drawContours(img, [hull], 0, (0,0,0), 1)
	#cv2.drawContours(img, [quad], 0, (0,0,0), 1)
	cv2.imshow("do",img)
#	cv2.waitKey(0)

	if(len(quad) != 4):
		return 0

	#DETERMINE CORNERS
	pts = quad.reshape(4, 2)
	rect = np.zeros((4, 2), dtype="float32")

	#FIND CORNERS OF BOUNDING RECT
	(tl_x,tl_y,w,h) = cv2.boundingRect(pts)
	tr_x = tl_x + w
	tr_y = tl_y
	bl_x = tl_x
	bl_y = tl_y + h
	br_x = tr_x
	br_y = tl_y + h

	#FIND TOP LEFT
	pyth = []
	for i in pts:
		pyth.append(np.sqrt((tl_x-i[0])**2 + (tl_y-i[1])**2))

	tl = pts[pyth.index(min(pyth))]
	pts = np.delete(pts,(np.where(np.all(pts == tl,axis=1))),0)

	#BOTTOM RIGHT FURTHEST FROM TOP LEFT
	pyth = []
	for i in pts:
		pyth.append(np.sqrt((tl[0]-i[0])**2 + (tl[1]-i[1])**2))

	br = pts[pyth.index(max(pyth))]
	pts = np.delete(pts,(np.where(np.all(pts == br,axis=1))),0)
	
	#REST ARE ???
	pyth = []
	for i in pts:
		pyth.append(np.sqrt((tr_x-i[0])**2 + (tr_y-i[1])**2))
	pyth_sort = sorted(pyth)
	
	tr = pts[pyth.index(pyth_sort[0])]
	bl = pts[pyth.index(pyth_sort[1])]

	rect = np.array([tl,tr,br,bl],dtype="float32")

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))

	#OUTPUT
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warp_img = cv2.warpPerspective(raw, M, (maxWidth,maxHeight))
	
	if(maxHeight <= 2*CROP_PX or maxWidth <= 2*CROP_PX):
		print("SMALL")
		return 0
	crop_img = warp_img[CROP_PX:maxHeight-CROP_PX,CROP_PX:maxWidth-CROP_PX].copy()
	
	if(maxWidth > maxHeight):
		warp_img = cv2.rotate(crop_img,cv2.ROTATE_90_CLOCKWISE).copy()
		maxHeight,maxWidth = maxWidth,maxHeight

	cv2.imshow("crop",warp_img)

	warp_flip_img = cv2.rotate(warp_img,cv2.ROTATE_180).copy()

	warp = Image.fromarray(warp_img)
	warp_flip = Image.fromarray(warp_flip_img)

	#FIND BEST MATCH
	key_min = ""
	stds = 0
	fkey = []
	fstd = []
	for i in range(0,2):
		if(i == 1):
			warp = warp_flip.copy()
		img_hash = np.packbits(imagehash.phash(warp, hash_size=32).hash.flatten())

		keys = []
		hams = []
		for key, value in d.items():
			h = hamming(img_hash,value)
			keys.append(key)
			hams.append(int(h))
		mini = min(hams)
		mean = np.mean(hams)
		std = np.std(hams)
		if(std == 0):
			continue

		key_min = keys[hams.index(mini)]
		fkey.append(key_min)
		
		stds = (mean-mini)/std
		fstd.append(stds)
	
	if(len(fstd) == 0):
			return 0
			
	if(max(fstd) > STDEVS):
		if(fstd[0] > fstd[1]):
			key_min = fkey[0]
			std = fstd[0]
		else:
			key_min = fkey[1]
			std = fstd[1]
	else:
		return 0
		#if(stds < STDEVS):
		#	if(i == 1):
		#		return 0
		#	continue
		#else:
		#	print(stds)
		#	break

	if(key_min == ""):
		return 0
	print(std)
	return(key_min)

#	print(stds,key_min)
#	f = cv2.imread("nrdb/"+key_min+".jpg")

cam = cv2.VideoCapture(0)
bak = None
last = None
refPt = []
cropping = False
showBox = True
invert = True

while(True):
	#GRAB GREYSCALE IMAGE
	cam.grab()
	ret_val, raw = cam.read()
	img = raw.copy()
	grey = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
	if(len(refPt) == 2 and cropping == False and showBox == True):
		cv2.rectangle(img, refPt[0], refPt[1], (0, 0, 255), 2)
	cv2.imshow("img",img)
	cv2.setMouseCallback("img", click_and_crop)

#	grey = grey[0:480, 50:450]

	#SUBTRACT BACKGROUND AND RUN
	key = 0
	if(bak is not None):
		sub = cv2.absdiff(grey,bak)
		if(len(refPt) == 2):
			sub = sub[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			#img = img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			grey = grey[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		if(invert):
			sub = ~sub
		key = identify(sub,grey)
	else:
		if(len(refPt) == 2):
			grey = grey[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			#img = img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		if(invert):
			sub = ~grey
		key = identify(sub,grey)

	#IF DIFFERENT FECTCH FROM NRDB
	if(key != 0 and key != last):
		print(key)
		#languages are en, zh-simp
		#req = urllib.request.urlopen('https://netrunnerdb.com/card_image/large/'+key+'.jpg')
		url = 'https://www.jinteki.net/img/cards/'+LANG+'/default/'+key+'.png'
		print("sending image request for "+url)
		try:
			req = urllib.request.urlopen(url)
		except:
			print("...404")
			if(LANG != 'en'):	#find english version
				url = 'https://www.jinteki.net/img/cards/en/default/'+key+'.png'
				print("trying for english version at "+url)
				try:
					req = urllib.request.urlopen(url)
				except:
					print("...404")
					(folder, num) = key.split("/")
					if(folder != 'stock'):	#find stock in preferred language
						url = 'https://www.jinteki.net/img/cards/'+LANG+'/default/stock/'+num+'.png'
						print("trying for stock version at "+url)
						try:
							req = urllib.request.urlopen(url)
						except:
							print("...404")
							continue
					else:
						continue
			else:
				(folder, num) = key.split("/")
				if(folder != 'stock'):	#find stock version
					url = 'https://www.jinteki.net/img/cards/'+LANG+'/default/stock/'+num+'.png'
					print("trying for stock version at "+url)
					try:
						req = urllib.request.urlopen(url)
					except:
						print("...404")
						continue
				else:
					continue
		print("...image received")
		#should probably check if image is successful
		if(LANG != 'en'):
			next
			#check if missing
			#if missing, switch to english
		
		arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
		f = cv2.imdecode(arr,-1)
		f = cv2.resize(f,(452,632))
		cv2.imshow("match",f)
		last = key
	
	c = cv2.waitKey(1)

	#KEY COMMANDS
	if c == 27:	# esc to quit
		break  
	elif c == 98:	# b to set background
		bak = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
	elif c == 99:	#c to clear background and box
		refPt = []
		bak = None
	elif c == 104: #h to hide box
		showBox = not showBox
	elif c == 105: #i to invert image
		invert = not invert