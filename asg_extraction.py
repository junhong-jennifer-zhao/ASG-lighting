import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches

import time
from PIL import Image

from scipy.ndimage.interpolation import zoom
import scipy.ndimage.filters as filters
from scipy import ndimage

from sklearn import mixture
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy import signal
from scipy.signal import argrelextrema

from skimage.feature import peak_local_max
from skimage.morphology import label
from scipy.ndimage.measurements import center_of_mass
from scipy.stats import rv_discrete

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from scipy.interpolate import Rbf
from scipy.special import comb

import cv2

import sys
import imageio as im

from multiprocessing import Pool
from itertools import repeat
import util
import ibl_roughmap as _rm

class ModelInfo():
	# ID, nParams
	GAUSS_SPHERE_SX 				= [0,1]
	GAUSS_SPHERE_SX_SY 				= [1,2]
	GAUSS_SPHERE_SX_SY_T 			= [2,3]

def sphericalToXY(v, l, denomenator):
	return (v - l) * denomenator

def lat2y(lat, height):
	latDen 	= (height-1)/np.pi
	y 	= sphericalToXY(np.pi/2, lat, latDen)
	if np.isscalar(y):
		return int(round(y))
	else:
		return np.round(y).astype(int)

def lon2x(lon, width):
	longDen = (width-1)/(np.pi*2);
	x 	= sphericalToXY(np.pi, lon, longDen)
	if np.isscalar(x):
		return int(round(x))
	else:
		return np.round(x).astype(int)

def ll2xy(lat,lon,width,height):
	return lon2x(lon,width), lat2y(lat,height)

def cartesian2latLon(x, y, z):
	lat = np.arcsin(y)
	lon = np.arccos(np.clip(z/np.cos(lat),-1,1)) * np.sign(x)
	return lat,lon

def cartesian2xy(x,y,z,w,h):
	lat,lon = cartesian2latLon(x,y,z)
	x,y = ll2xy(lat,lon,w,h)
	return x,y
	
def latLon2Cartesian(lat, lon, r=1):
	x = r * np.sin(lon) * np.cos(lat)
	y = r * np.sin(lat)
	z = r * np.cos(lon) * np.cos(lat)

	if not np.isscalar(x):
		y = np.repeat(y, x.shape[1], axis=1)

	return np.asarray([x,y,z])

def xyToSpherical(v, numerator, denomenator):
	return v - (numerator/denomenator)

def yLocToLat(yLoc, height):
	latDen 	= (height-1)/np.pi
	lat 	= xyToSpherical(np.pi/2, yLoc%height, latDen)
	return lat

def xLocToLon(xLoc, width):
	longDen = (width-1)/(np.pi*2);
	lon 	= xyToSpherical(np.pi, xLoc%width, longDen)
	return lon

def xy2ll(x,y,width,height):
	return yLocToLat(y, height), xLocToLon(x, width)
	#return np.asarray([yLocToLat(y, height), xLocToLon(x, width)])

def xy2XYZ(x,y,width,height):
	latLon = xy2ll(x,y,width,height)
	return latLon2Cartesian(latLon[0], latLon[1])

def crossNormalised(x,y):
	vec = np.cross(x,y)
	return vec/np.linalg.norm(vec)

def removeBottom(img):
	#rint(img.shape)
	height = img.shape[0]
	if len(img.shape)==3:
		return img[:int(height/2),:,:]
	return img[:int(height/2),:]

def loadData(fn, width):
	img = im.imread(fn)

	height = int(width/2)
	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	return img

def gaussian2D_v3(xSize,ySize, amp=1.0,stdx=1,stdy=1,theta=0.0,center=None):
	x = np.arange(0, xSize, 1, float)
	y = np.arange(0, ySize, 1, float).reshape(ySize,1)
	if center is None:
		x0 = xSize // 2
		y0 = ySize // 2
	else:
		x0 = center[0]
		y0 = center[1]

	stdx2 = (2*stdx)**2
	stdy2 = (2*stdy)**2

	stdx4 = (4*stdx)**2
	stdy4 = (4*stdy)**2

	a = ((np.cos(theta)**2) / stdx2) + ((np.sin(theta)**2) / stdy2) 
	b = (-((np.sin(2*theta)) / stdx4)) + ((np.sin(2*theta)) / stdy4)
	c = ((np.sin(theta)**2) / stdx2) + ((np.cos(theta)**2) / stdy2) 
	
	return  amp*np.exp( - (a*(x-x0)**2 - 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def gaussianSphere(xyz, width, height, center, amp=1.0,stdx=1,stdy=1,theta=0.0):
	def evaluateSG(v):
		# http://cg.cs.tsinghua.edu.cn/people/~kun/asg/paper_asg.pdf
		z = xy2XYZ(center[0],center[1],width,height)
		x = crossNormalised(z,np.asarray([0,1,0]))
		y = crossNormalised(z,x)
		def rotate(v, axis, theta):
			return v*np.cos(theta) + crossNormalised(axis,v) * np.sin(theta) + axis*np.dot(axis,v)*(1.0-np.cos(theta))
		x = rotate(x,z,theta)
		y = rotate(y,z,theta)
		n = 2
		distribution = np.maximum(np.sum(v*z, axis=2),0) * np.exp( -((np.sum(v*x, axis=2)**n)/stdx) - ((np.sum(v*y, axis=2)**n)/stdy) )
		return amp * distribution[:,:,np.newaxis]
	return evaluateSG(xyz)

def getWeights(iblWidth):
	iblHeight = int(iblWidth/2)
	x = np.arange(0,iblWidth)
	y = np.arange(0,iblHeight).reshape(iblHeight,1)
	latitudeArray = (xy2ll(x,y,iblWidth,iblHeight)[0]).reshape(-1)
	weightsArray = np.cos(latitudeArray)
	weightsImage = np.repeat(weightsArray[:, np.newaxis], iblWidth, axis=1)
	#weightsImage = removeBottom(weightsImage)
	#plt.imshow(weightsImage)
	#plt.show()
	#sys.exit()
	return weightsImage

def resizeImg2(img, width, height, interpolation=cv2.INTER_CUBIC):
	if img.shape[1]<width: # up res
		if interpolation=='max_pooling':
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
		else:
			return cv2.resize(img, (width, height), interpolation=interpolation)
	if interpolation=='max_pooling': # down res, max pooling
		try:
			import skimage.measure
			scaleFactor = int(float(img.shape[1])/width)
			factoredWidth = width*scaleFactor
			img = cv2.resize(img, (factoredWidth, int(factoredWidth/2)), interpolation=cv2.INTER_CUBIC)
			blockSize = scaleFactor
			r = skimage.measure.block_reduce(img[:,:,0], (blockSize,blockSize), np.max)
			g = skimage.measure.block_reduce(img[:,:,1], (blockSize,blockSize), np.max)
			b = skimage.measure.block_reduce(img[:,:,2], (blockSize,blockSize), np.max)
			img = np.dstack((np.dstack((r,g)),b)).astype(np.float32)
			return img
		except:
			print("Failed to do max_pooling, using default")
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	else: # down res, using interpolation
		return cv2.resize(img, (width, height), interpolation=interpolation)

def resizeImg(img, w, h):
	return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

def makeBinaryMaskFromIndices(shape, coords):
	binaryMask = np.zeros(shape, dtype=int)
	for coord in coords:
		y, x = coord
		y = int(y)
		x = int(x)
		binaryMask[y,x] = 1
	return binaryMask

def weighted_avg_and_std(values, weights):
	weights = weights[:values.shape[0],:] # incase the image is cropped (e.g bottom hemisphere removed)
	if len(values.shape)==3:
		average = np.average(values, weights=weights, axis=(0, 1))
		variance = np.average((values-average)**2, weights=weights, axis=(0, 1))
	else:
		average = np.average(values, weights=weights)
		variance = np.average((values-average)**2, weights=weights)
	return (average, np.sqrt(variance))

def poleScale(y, width, relative=True):
	"""
	y = y pixel position (cast as a float)
	Scaling pixels lower toward the poles
	Sample scaling in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scalefactors.pdf
	"""
	height = int(width/2)
	piHalf = np.pi/2
	pi4 = np.pi*4
	pi2OverWidth = (np.pi*2)/width
	piOverHeight = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	scaleFactor = (1.0 / pi4) * pi2OverWidth * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))
	if relative:
		scaleFactor /= (1.0 / pi4) * pi2OverWidth * (np.cos(piHalf - (piOverHeight / 2.0)) - np.cos(piHalf + (piOverHeight / 2.0)))
	return scaleFactor

def getPoleScaleMap(width):
	height = int(width/2)
	return np.repeat(poleScale(np.arange(0,height), width)[:, np.newaxis], width, axis=1)

def toGrey(x):
	if len(x.shape)>2:
		return (x[:,:,0]+x[:,:,1]+x[:,:,2])
	return x

def detectAllLights(imgInput, max_sigma=30, threshold=None, nStdAboveMean=0.0, blurAmount=5, upperHemisphereOnly=False):
	'''
	Returns local maximas
	Each row contains [y pixel coord, x pixel coord, size, intensity]
	'''
	#
	mode='equirectangular'
	max_sigma = 0.05*float(imgInput.shape[1])
	img = np.copy(imgInput)
	nRows = img.shape[0]
	nCols = img.shape[1]
	poleWeights = getPoleScaleMap(img.shape[1])

	# Process image
	if blurAmount > 0:
		img = cv2.GaussianBlur(img, (blurAmount, blurAmount), 0)
	img = img/np.max(img) # normalize

	if len(img.shape) > 2:
		img = toGrey(img)

	if threshold is None:
		m1,s1 = weighted_avg_and_std(img,poleWeights)
		#threshold = np.mean(img*poleWeights) + (np.std(img*poleWeights)*nStdAboveMean)
		threshold = m1+(s1*nStdAboveMean)
		#threshold = 0.1

	if mode=="equirectangular": # 360 image repeat edge
		rollAmountX = int(nCols/2)
		img = np.hstack((img,img)) # side by side
		img = np.vstack((np.flip(img,axis=0),img)) # top bottom
		#img = np.vstack((img,np.flip(img,axis=0))) # bottom top

		img = np.roll(img, rollAmountX, axis=1) # centre the image x
		poleWeights = getPoleScaleMap(img.shape[1])

	# coord detect
	print("threshold %.4f" % threshold)
	#print("max_sigma %.4f" % max_sigma)

	def detectPeaks(img, max_sigma, threshold):
		peakCoords = peak_local_max(img, min_distance=3, threshold_abs=0)
		if peakCoords.shape[0]==0:
			return peakCoords
		binaryMask = makeBinaryMaskFromIndices((img.shape[0],img.shape[1]), peakCoords)
		# Remove plateaus
		doPlateasCheck = True
		if doPlateasCheck:
			label_image = label(binaryMask)
			finalCoordinates = []
			for value in np.unique(label_image):
				if value==0: # Skip the background
					continue
				# Set the label to 1, everything else 0
				oneLabelImage = np.copy(label_image)
				oneLabelImage[oneLabelImage!=value] = 0
				oneLabelImage[oneLabelImage>0] = 1

				# If there's only one pixel, let's just use it's coordinate using np.where
				nCoordinatesForValue = np.count_nonzero(oneLabelImage)
				if nCoordinatesForValue==1:
					singlePixelCoordinate = np.where(oneLabelImage>0)
					finalCoordinates.append([int(singlePixelCoordinate[0]),int(singlePixelCoordinate[1])])		
					continue
				elif nCoordinatesForValue==0: # If for whatever reason we have no coordinates, let's skip
					continue
				# Finally. we have a plateau, so use centre of mass (polescale to account for spherical distorition in latlon image)
				oneLabelImage = oneLabelImage.astype(np.float32) * poleWeights
				CoM = center_of_mass(oneLabelImage)
				finalCoordinates.append([int(CoM[0]),int(CoM[1])])
			peakCoords = np.asarray(finalCoordinates)

		sizes = np.asarray([1]*peakCoords.shape[0]).reshape((peakCoords.shape[0],1))*0.3
		peakCoords = np.hstack((peakCoords,sizes))

		# Remove under threshold
		doThresholdCheck = True
		if doThresholdCheck:
			badCounts=[]
			count=0
			for coord in peakCoords:
				y, x, r = coord
				y = int(y)
				x = int(x)
				if img[y,x]<threshold:
					badCounts.append(count)
				count+=1
			if len(badCounts)>0:
				peakCoords = np.delete(peakCoords, badCounts, axis=0)
		return peakCoords

	peakCoords = detectPeaks(img, max_sigma, threshold)
	if len(peakCoords)==0:
		print("WARNING 2: No peaks detected. Will use max instead.")
		return np.asarray(peakCoords)

	if mode=="equirectangular": # Account for 360 image repeating edge and centring
		peakCoords[:, 0] -= (nRows-1)
		peakCoords[:, 1] -= rollAmountX
		validX = np.logical_and(peakCoords[:,1] >= 0, peakCoords[:,1] < nCols )
		peakCoords = peakCoords[validX,:]
		validY = np.logical_and(peakCoords[:,0] >= 0, peakCoords[:,0] < nRows )
		peakCoords = peakCoords[validY,:]
	
	# horizon line
	if upperHemisphereOnly:
		badCounts=[]
		count=0
		for coord in peakCoords:
			y, x, r = coord
			y = int(y)
			x = int(x)
			horizonLineCheck = y > (nRows/2)
			if horizonLineCheck:
				badCounts.append(count)
			count+=1
		if len(badCounts)>0:
			peakCoords = np.delete(peakCoords, badCounts, axis=0)

	# Add the intensities of each local maxima
	peakCoordsInt = peakCoords.astype(int)
	intensities = img[peakCoordsInt[:,0],peakCoordsInt[:,1]]
	peakCoords = np.hstack((peakCoords,intensities.reshape(intensities.shape[0],1)))

	# Sort them 
	peakCoords = peakCoords[peakCoords[:,-1].argsort()[::-1]] # highest to lowest

	return peakCoords,threshold #

def getXYZData(w,h):
	x = np.arange(0,w)
	y = np.arange(0,h).reshape(h,1)
	latLon = xy2ll(x,y,w,h)
	cart = xy2XYZ(x,y,w,h)
	return np.swapaxes(np.array(list(zip(cart[0],cart[1],cart[2]))),1,2)

def getGuess(nLights,nParams,resReduction):
	guess = []
	geussSigma = (0.0174533)*resReduction
	thetaStart=0.0
	#nLights = 1
	for i in range(0, nLights):
		if nParams==1:
			guess += [geussSigma]
		elif nParams==2:
			guess += [geussSigma,geussSigma]
		elif nParams==3:
			guess += [geussSigma,geussSigma,thetaStart]
		elif nParams==5:
			guess += [geussSigma,geussSigma,thetaStart,geussSigma,geussSigma]
	return guess

def processLightDetection(targetImgRGB,nStdAboveMean):
	targetImg = (targetImgRGB[:,:,0]+targetImgRGB[:,:,1]+targetImgRGB[:,:,2])/3

	# Light detection
	coords_img,threshold = detectAllLights(targetImg, threshold=None, nStdAboveMean=nStdAboveMean, blurAmount=0, upperHemisphereOnly=False) # overdetect ground truth (get anything that looks like a light)
	coords_img = coords_img.astype(int)
	X = coords_img[:,1].astype(int)
	Y = coords_img[:,0].astype(int)
	print(X,Y)
	lightIndices = np.where(targetImg[Y,X] > 0.0)[0]

	nLights = len(lightIndices)
	print('Num Lights: %d' % nLights)
	print("Light detection complete")

	# Ambient light value
	poleWeights = getPoleScaleMap(targetImg.shape[1])
	poleWeights[targetImg>=threshold] = 0
	ambientLightValue,_ = weighted_avg_and_std(targetImgRGB,np.dstack((poleWeights,poleWeights,poleWeights)))
	print("ambientLightValue:",ambientLightValue)

	return X,Y,lightIndices,nLights,threshold,ambientLightValue


def reduceResolution(iblWidth,iblHeight,oldWidth,img,imgBlur,X,Y,lightIndices):
	imgLights = np.copy(img)
	# Resolution reduction
	img		= resizeImg(img, iblWidth, int(iblHeight))
	imgBlur = resizeImg(imgBlur, iblWidth, int(iblHeight))

	# 
	imgLights = resizeImg(imgLights, iblWidth, int(iblHeight))
	resReduction = (float(iblWidth)/oldWidth)
	print("Resolution reduction: %f" % (resReduction))
	X = (X*resReduction-1).astype(int)
	Y = (Y*resReduction-1).astype(int)
	
	# Draw lights
	imgLights /= np.max(imgLights)
	imgLights[Y[lightIndices],X[lightIndices],1] = 10000.0
	print("Resolution reduction complete")

	return img,imgBlur,X,Y,resReduction

def process_samples(vec1,img):
	npoints = vec1.shape[1]
	cols = []
	values = []
	vec1 /= np.linalg.norm(vec1, axis=0)
	vec2 = np.copy(vec1)
	maxVal = np.max(img)
	if img is not None:
		w = img.shape[1]
		h = img.shape[0]
		for i in range(0,npoints):
			x,y,z = vec2[0,i], vec2[1,i], vec2[2,i]
			lat,lon = cartesian2latLon(x,y,z)
			px,py = ll2xy(lat,lon,w,h)
			pixel = img[py,px,:]
			cols.append(list(np.clip(np.log1p(pixel),0,1)))

			scale = np.log1p(np.max(pixel))
			values.append(np.max(pixel))
			vec2[:,i] *= scale
	return vec1.T, vec2.T, cols, np.asarray(values), npoints

def sample_spherical(npoints, img=None):
	vec1 = np.random.randn(3,npoints)
	return process_samples(vec1,img)

def sample_spherical_biased(npoints, img, nStdAboveMean=0.0, doRemoveBottom=False):
	greyImg = np.max(img,axis=2)
	if nStdAboveMean is not None:
		greyImg[greyImg<np.mean(greyImg)+np.std(greyImg)*nStdAboveMean] = 0
	if doRemoveBottom:
		greyImg[int(greyImg.shape[0]/2):,:] = 0

	greyImg[greyImg>0] = 1

	# Weighted sampling
	distribution = (greyImg/np.sum(greyImg)).reshape(-1)
	imgIndices = np.arange(distribution.shape[0])
	random_variable = rv_discrete(values=(imgIndices,distribution))
	sample_indices = random_variable.rvs(size=npoints)

	# Throw away duplicates
	sample_indices = np.unique(sample_indices)
	npoints = sample_indices.shape[0]
	
	Y,X = np.unravel_index(sample_indices, (img.shape[0],img.shape[1]))
	coords = []
	for i in range(0,npoints):
		coords.append(xy2XYZ(X[i],Y[i],img.shape[1],img.shape[0]))
	vec1 = np.asarray(coords).T
	return process_samples(vec1,img)

def normalize(x):
	if np.dot(x,x)==0:
		return x
	return x/np.linalg.norm(x)

def slerp_points(a,b,nPointsPerDegree=1,showPlot=False,axis_order=[0,2,1]):
	# Computes N spherically interpolated points between two 3D points (a and b)
	# where N scales based on the angle between a and b using nPointsPerDegree.
	# a and b are normalized.
	"""
	Computes N spherically interpolated points between two 3D points (a and b)
	where N scales based on the angle between a and b using nPointsPerDegree.
	Parameters
	----------
	a : ndarray (1,3)
		The point to slerp from.
	b : ndarray (1,3)
		The point to slerp to.
	nPointsPerDegree : float, optional
		The number of points to generate per degree between a and b.
	showPlot : bool, optional
	    Whether to display the points in a 3D plot.
	Returns
	-------
	points : ndarray (N,3)
		Returns (N,3) shape, where each row is an slerped point between a and b.
	Examples
	--------
	points = slerp_points(np.asarray([0,1,0]), np.asarray([1,0,0]),0.5,True)
	"""

	# Helper function
	def normalize(x):
		if np.dot(x,x)==0: # avoid origin
			return x
		return x/np.linalg.norm(x)

	# avoid origin
	if np.dot(a,a)==0:
		return np.asarray([b,])
	if np.dot(b,b)==0:
		return np.asarray([a,])

	# normalize
	a = normalize(a)
	b = normalize(b)

	# Make perpendicular vector
	if np.abs(np.dot(a,b))>=1.0: 
		# account for parralel vectors: be careful of the direction of the arc
		if a[1]!=1:
			k = normalize(np.cross(a,[0,1,0]))
		else:
			k = normalize(np.cross(a,[1,0,0]))
	else:
		k = normalize(np.cross(a,b))

	# Angle between a and b
	theta = np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))
	
	# Generate angles, precompute sin and cos
	N = int((theta*57.2958)*nPointsPerDegree)
	if N<=0:
		N = 1
	angles = np.linspace(0,theta,N)
	angles_cos = np.cos(angles)[:,None]
	angles_sin = np.sin(angles)[:,None]

	# Tile data for vectorisation
	a_tiled = np.tile(a, (N,1))
	k_tiled = np.tile(k, (N,1))
	kXa_tiled = np.tile(np.cross(k,a), (N,1))
	kDa_tiled = np.tile(np.dot(k,a), (N,1))

	# Compute N angles between a and b
	points = a_tiled*angles_cos+kXa_tiled*angles_sin+k_tiled*kDa_tiled*(1.0-angles_cos)

	if showPlot:
		fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal', 'facecolor':'gray'})
		ax.scatter(0,0,0, s=10, c='k')
		ax.scatter(a[axis_order[0]], a[axis_order[1]], a[axis_order[2]], s=10, c='r')
		ax.scatter(b[axis_order[0]], b[axis_order[1]], b[axis_order[2]], s=10, c='g')
		ax.scatter(points[:,axis_order[0]], points[:,axis_order[1]], points[:,axis_order[2]], s=1, c='b')
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		plt.show()

	return points

def smooth(y, box_pts=3):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	y_smooth[0] = y_smooth[1] + (y_smooth[1]-y_smooth[2])
	y_smooth[-1] = y_smooth[-2] + (y_smooth[-2]-y_smooth[-3])
	return y_smooth


def climbLightDistance6(y_raw, showPlot=False):
	y = y_raw[::-1]/np.max(y_raw) # reverse array (for gaussian fit), and normalize (for relative error)
	x = np.arange(0,y.shape[0])
	mu = 0 #x[-1]
	offset = 0.0 # y[-1] # offset ensure gaussian end is at minimum the y data (note: it can appear higher if sigma is large)
	scale = y[0]-offset

	f_estimate = np.clip(y[-1],0.0001,0.9999) # to avoid NaN
	sigma_guess = np.sqrt(np.abs(((x[-1]*x[-1])/(4*np.log((f_estimate-offset)/scale)))))

	def gaussian_func(x, sigma):
		return offset + (scale*np.exp(-(x-mu)**2/(2.0*sigma**2)))

	try:
		#coeff = curve_fit(f=gaussian_func, xdata=x, ydata=y, p0=[sigma_guess], maxfev=5000)[0] #, ftol=0.1, xtol=0.1)[0] #, bounds=((0), (np.inf)))[0]
		coeff = curve_fit(f=gaussian_func, xdata=x, ydata=y, p0=[sigma_guess], maxfev=5000)[0] #, bounds=((0), (np.inf)))[0]
	except:
		coeff = [sigma_guess]
		print("No optimal parameters found, using default:", sigma_guess)

	y_fit = gaussian_func(x, *coeff)
	distance = np.sum(np.square(y-y_fit)) #+ np.sum(y<threshold)
	if showPlot:
		print(distance, ':', offset, scale)
		plt.plot(x, y, c='k')
		plt.plot(x, y_fit, c='r', ls='--')
		plt.plot(x, gaussian_func(x, *[sigma_guess]), c='c', ls='--')
		plt.show()
	return distance

def climbLightDistance7(y_raw, showPlot=False):
	y = y_raw[::-1]/np.max(y_raw) # reverse array (for gaussian fit), and normalize (for relative error)
	x = np.arange(0,y.shape[0])
	offset = 0.0 # y[-1] # offset ensure gaussian end is at minimum the y data (note: it can appear higher if sigma is large)
	scale = y[0]-offset
	mu = 0 #x[-1]
	f_estimate = np.clip(y[-1],0.0001,0.9999) # to avoid NaN
	sigma_guess = np.sqrt(np.abs(((x[-1]*x[-1])/(4*np.log((f_estimate-offset)/scale)))))

	def gaussian_func(x, *p):
		sigma, n = p
		return offset+(scale*np.exp(-(x-mu)**n/(2.0*sigma**n)))
	try:
		coeff = curve_fit(f=gaussian_func, xdata=x, ydata=y, p0=[sigma_guess,2.0], maxfev=5000, bounds=((0), (np.inf)))[0]
		sigma = coeff[0]
		n = coeff[1]
	except:
		coeff = [sigma_guess,2.0]
		sigma = coeff[0]
		n = coeff[1]
		print("No optimal parameters found, using default:", sigma_guess, 2.0)

	y_fit = gaussian_func(x, *coeff)
	distance = np.sum(np.square(y-y_fit))
	if showPlot:
		print(distance, ':', offset, scale, coeff, sigma_guess, sigma, n)
		plt.plot(x, y, c='k')
		plt.plot(x, y_fit, c='r', ls='--')
		plt.plot(x, gaussian_func(x, *[sigma_guess,2]), c='c', ls='--')
		plt.show()
	return distance

def climbLightDistance_Analytical(y_raw, threshold=0.0, showPlot=False):
	y = y_raw[::-1]/np.max(y_raw) # reverse array (for gaussian fit), and normalize (for relative error)
	x = np.arange(0,y.shape[0])
	mu = 0 #x[-1]
	offset = 0.0 # y[-1] # offset ensure gaussian end is at minimum the y data (note: it can appear higher if sigma is large)
	scale = y[0]-offset
	f_estimate = np.clip(y[-1],0.0001,0.9999) # to avoid NaN
	sigma_guess = np.sqrt(np.abs(((x[-1]*x[-1])/(4*np.log((f_estimate-offset)/scale)))))

	def gaussian_func(x, sigma):
		return (scale*np.exp(-(x-mu)**2/(2.0*sigma**2)))

	coeff = curve_fit(f=gaussian_func, xdata=x, ydata=y, p0=[sigma_guess], maxfev=5000)[0] #
	y_fit = gaussian_func(x, *coeff)
	distance = np.sum(np.square(y-y_fit)) #
	if showPlot:
		print(distance, ':', offset, scale, coeff, sigma_guess)
		plt.plot(x, y, c='k')
		plt.plot(x[y<threshold], y[y<threshold], c='r')
		plt.plot(x, y_fit, c='r', ls='--')
		plt.plot(x, gaussian_func(x, sigma_guess), c='c', ls='--')
		plt.show()
	return distance

def climbLightDistance_spline(y_raw, threshold=0.0, showPlot=False):
	y = y_raw[::-1]/np.max(y_raw) # reverse array (for gaussian fit), and normalize (for relative error)
	x = np.linspace(0,1,y.shape[0])

	points = np.vstack((x,y)).T

	Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*comb(n,k)
	bezierM = lambda ts: np.matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])
	def lsqfit(points,M):
		M_ = np.linalg.pinv(M)
		return M_ * points
	M = bezierM(x)
	control_points=lsqfit(points, M)
	control_points[0,0] = 0
	control_points[0,1] = y[0]
	control_points[3,0] = x[-1]
	control_points[3,1] = np.min(y) #y[-1]

	y_fit = M*control_points
	distance = np.sum(np.square(points-y_fit))
	if showPlot:
		plt.plot(points[:,0],points[:,1], c='r')
		plt.scatter([control_points[:,0]],[control_points[:,1]],c='g')
		plt.plot(y_fit[:,0], y_fit[:,1], c='b')
		plt.show()

	return distance

def curveClustering(data,k,img,dist_func):
	plt.clf()

	img /= np.max(img)
	nSamples = data.shape[0]

	width = img.shape[1]
	height = img.shape[0]

	displayImg = np.dstack((img,img,img))

	test_x = -1 
	test_y = -1 

	print("Test coords:", test_x, test_y)

	startTime = time.time()
	clustering_labels = np.ones((nSamples),np.int_)*-1
	for lightIdx in range(0,k):
		clustering_labels[lightIdx] = lightIdx

	plt.figure(1)
	for sampleIdx in range(k,nSamples):
		min_distance = np.inf
		min_lightIdx = -1
		coord_sample = data[sampleIdx,:]
		coord_sample = coord_sample/np.linalg.norm(coord_sample)
		x0,y0 = cartesian2xy(coord_sample[0],coord_sample[1],coord_sample[2],width,height)
		for lightIdx in range(0,k): 
			coord_light = data[lightIdx,:]
			coord_light = coord_light/np.linalg.norm(coord_light)

			angle_dist = np.clip(np.arccos(np.dot(coord_sample,coord_light)) / np.pi,0,1)

			x1,y1 = cartesian2xy(coord_light[0],coord_light[1],coord_light[2],width,height)
			points = slerp_points(coord_sample,coord_light,2,False)
			x,y = cartesian2xy(points[:,0],points[:,1],points[:,2],width,height)
			zi = img[y.astype(np.int_), x.astype(np.int_)]
			angle_distance = np.linalg.norm(coord_sample-coord_light)

			if x0==test_x and y0==test_y:
				print("Display img...")
				y = np.vstack((y,y-1,y-1,y+1,y+1))
				x = np.vstack((x,x,x,x,x))
				displayImg[y.astype(np.int_), x.astype(np.int_),0] = 0
				displayImg[y.astype(np.int_), x.astype(np.int_),1] = 1000
				displayImg[y.astype(np.int_), x.astype(np.int_),2] = 0
				plt.imshow(displayImg)
				#plt.show()
				displayImg[y.astype(np.int_), x.astype(np.int_),0] = 1000
				displayImg[y.astype(np.int_), x.astype(np.int_),1] = 0
				displayImg[y.astype(np.int_), x.astype(np.int_),2] = 0

			if zi.shape[0]<=3: # sample is close enough to light
				distance = 0
			else: # measure distance of sample from light
				zi_s = smooth(zi, 3)
				if x0==test_x and y0==test_y:
					#print(zi_s)
					plt.show()
					print('->',angle_distance)
					distance = dist_func(zi_s, showPlot=True)
					#print(distance)
				else:
					distance = dist_func(zi_s)

			distance += (angle_distance*0.05)
			if distance < min_distance:
				min_distance = distance
				min_lightIdx = lightIdx
		clustering_labels[sampleIdx] = min_lightIdx

	labellingTime = time.time()-startTime
	print('time (labels):',labellingTime)
	return clustering_labels

def curve_fitting_parralel_func(sampleIdx, k, data, img):
	min_distance = np.inf
	min_lightIdx = -1
	coord_sample = data[sampleIdx,:]
	coord_sample = coord_sample/np.linalg.norm(coord_sample)
	x0,y0 = cartesian2xy(coord_sample[0],coord_sample[1],coord_sample[2],img.shape[1],img.shape[0])
	for lightIdx in range(0,k): 
		coord_light = data[lightIdx,:]
		coord_light = coord_light/np.linalg.norm(coord_light)
		angle_dist = np.clip(np.arccos(np.dot(coord_sample,coord_light)) / np.pi,0,1)
		x1,y1 = cartesian2xy(coord_light[0],coord_light[1],coord_light[2],img.shape[1],img.shape[0])
		points = slerp_points(coord_sample,coord_light,2,False)
		x2,y2 = cartesian2xy(points[:,0],points[:,1],points[:,2],img.shape[1],img.shape[0])
		zi = img[y2.astype(np.int_), x2.astype(np.int_)]
		angle_distance = np.linalg.norm(coord_sample-coord_light)

		distance = 0
		if zi.shape[0]<=3: # sample is close enough to light
			distance = 0
		else: # measure distance of sample from light
			zi_s = np.convolve(zi, np.ones(3)/3, mode='same')
			zi_s[0] = zi_s[1] + (zi_s[1]-zi_s[2])
			zi_s[-1] = zi_s[-2] + (zi_s[-2]-zi_s[-3])

			################
			y = zi_s[::-1]/np.max(zi_s) # reverse array (for gaussian fit), and normalize (for relative error)
			x = np.arange(0,y.shape[0])
			offset = 0.0 # y[-1] # offset ensure gaussian end is at minimum the y data (note: it can appear higher if sigma is large)
			scale = y[0]-offset
			mu = 0 #x[-1]
			f_estimate = np.clip(y[-1],0.0001,0.9999) # to avoid NaN
			sigma_guess = np.sqrt(np.abs(((x[-1]*x[-1])/(4*np.log((f_estimate-offset)/scale)))))
			def gaussian_func(x, *p):
				sigma, n = p
				return offset+(scale*np.exp(-(x-mu)**n/(2.0*sigma**n)))
			try:
				#coeff = curve_fit(f=gaussian_func, xdata=x, ydata=y, p0=[sigma_guess,2.0], maxfev=5000)[0] #, ftol=0.1, xtol=0.1)[0] #, bounds=((0), (np.inf)))[0]
				coeff = curve_fit(f=gaussian_func, xdata=x, ydata=y, p0=[sigma_guess,2.0], maxfev=5000, bounds=((0), (np.inf)))[0]
				sigma = coeff[0]
				n = coeff[1]
			except:
				coeff = [sigma_guess,2.0]
				sigma = coeff[0]
				n = coeff[1]
				print("No optimal parameters found, using default:", sigma_guess, 2.0)
			distance = np.sum(np.square(y-gaussian_func(x, *coeff)))
		distance += (angle_distance*0.05)
		if distance < min_distance:
			min_distance = distance
			min_lightIdx = lightIdx
	return min_lightIdx

def curveClusteringParralel(data,k,img):
	img /= np.max(img)
	nSamples = data.shape[0]
	startTime = time.time()
	clustering_labels = np.ones((nSamples),np.int_)*-1
	for lightIdx in range(0,k):
		clustering_labels[lightIdx] = lightIdx
	with Pool() as pool:
		min_lightIndices = pool.starmap(curve_fitting_parralel_func, zip(range(k,nSamples), repeat(k), repeat(data), repeat(img)))
	clustering_labels[k:] = min_lightIndices
	labellingTime = time.time()-startTime
	print('time (labels):',labellingTime)
	return clustering_labels

def euclideanClustering(data,k):
	nSamples = data.shape[0]
	startTime = time.time()
	clustering_labels = np.ones((nSamples),np.int_)*-1
	for lightIdx in range(0,k):
		clustering_labels[lightIdx] = lightIdx
	for sampleIdx in range(k,nSamples):
		#print(sampleIdx)
		min_distance = np.inf
		min_lightIdx = -1
		coord_sample = data[sampleIdx,:]
		for lightIdx in range(0,k):
			coord_light = data[lightIdx,:]
			distance = np.linalg.norm(coord_sample-coord_light)
			#distance = nx.astar_path_length(T, lightIdx,sampleIdx)
			#distance = nx.dijkstra_path_length(T, lightIdx,sampleIdx)
			#print(distance)
			if distance < min_distance:
				min_distance = distance
				min_lightIdx = lightIdx
		clustering_labels[sampleIdx] = min_lightIdx
	print('time (labels):',time.time()-startTime)
	return clustering_labels

def saveClusterImg(nSamples, labels_, data_samples_s1, imgBlur_full, cmap, k='', nSamplesInput='', clusteringMethod='', method_id=-1):
	clusterImg = np.zeros(imgBlur_full.shape,np.float32)
	clusterImg2 = np.log1p(np.copy(imgBlur_full))
	scaleFactor = np.max(clusterImg2)
	size = 2
	for i in range(0,nSamples):
		coord = data_samples_s1[i,:]
		coord = coord/np.linalg.norm(coord)
		clusters_pixelcoords = cartesian2xy(coord[0], coord[1], coord[2],width,height)
		color = cmap(labels_[i])
		clusterImg[clusters_pixelcoords[1],clusters_pixelcoords[0],0] = color[0]*1
		clusterImg[clusters_pixelcoords[1],clusters_pixelcoords[0],1] = color[1]*1
		clusterImg[clusters_pixelcoords[1],clusters_pixelcoords[0],2] = color[2]*1

		clusterImg2[clusters_pixelcoords[1],clusters_pixelcoords[0],0] = color[0]*scaleFactor
		clusterImg2[clusters_pixelcoords[1],clusters_pixelcoords[0],1] = color[1]*scaleFactor
		clusterImg2[clusters_pixelcoords[1],clusters_pixelcoords[0],2] = color[2]*scaleFactor
		if i < nLights:
			starty = clusters_pixelcoords[1]-size
			endy = clusters_pixelcoords[1]+size
			startx = clusters_pixelcoords[0]-size
			endx = clusters_pixelcoords[0]+size
			clusterImg[starty:endy,startx:endx,0] = color[0]*1
			clusterImg[starty:endy,startx:endx,1] = color[1]*1
			clusterImg[starty:endy,startx:endx,2] = color[2]*1
			clusterImg2[starty:endy,startx:endx,0] = color[0]*scaleFactor
			clusterImg2[starty:endy,startx:endx,1] = color[1]*scaleFactor
			clusterImg2[starty:endy,startx:endx,2] = color[2]*scaleFactor

def getPCA(nLights, data_samples_s1, labels_, cmap, n_components = 2, showPlot_PCA = False):
	pca_primary_components = [] 
	pca_stds = [] 
	pca = PCA(n_components=n_components)

	def rotate(v, axis, theta):
		return v*np.cos(theta) + crossNormalised(axis,v) * np.sin(theta) + axis*np.dot(axis,v)*(1.0-np.cos(theta))
	upVector = np.asarray([0,1,0])

	np.set_printoptions(precision=6)
	np.set_printoptions(suppress=True)
	fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto', 'facecolor':'gray'})
	for i in range(0,nLights):
		cluster_i = data_samples_s1[labels_==i,:]
		print(i, nLights, cluster_i.shape)

		if cluster_i.shape[0]<=1: # invalid size of cluster
			pca_primary_components.append(upVector)
			pca_stds.append([0.0000001,0.0000001])
			continue

		result = pca.fit(cluster_i)
		angle = (float(X[i])/width-0.5)*(np.pi*2)
		primary_component = result.components_[0]
		transformed_component = rotate(primary_component,upVector,angle)

		pca_primary_components.append(transformed_component)
		pca_stds.append(result.explained_variance_)

		if showPlot_PCA:
			ax.scatter(cluster_i[:,0], cluster_i[:,2], cluster_i[:,1], s=10, c=cmap(i), alpha=0.2, edgecolors=None)
			quiver_cols = ['r', 'g', 'b']
			for component in range(0,n_components):
				ax.quiver(cluster_i[0,0],cluster_i[0,2],cluster_i[0,1], result.components_[component,0],result.components_[component,2],result.components_[component,1], length=np.sqrt(result.explained_variance_[component]), color=quiver_cols[component])
	if showPlot_PCA:
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		plt.show()

	return pca_primary_components, pca_stds

def getGuessFromPCA(nLights,nParams,resReduction,pca_primary_components,pca_stds):
	guess = []
	thetas = []
	refVector = np.asarray([-1,0,0])
	x1 = refVector[0]
	y1 = refVector[1]
	z1 = refVector[2]
	for i in range(0, nLights):
		targetVector = pca_primary_components[i]
		# angle
		x2 = targetVector[0]
		y2 = targetVector[1]
		z2 = targetVector[2]
		dot = x1*x2 + y1*y2 + z1*z2
		lenSq1 = x1*x1 + y1*y1 + z1*z1
		lenSq2 = x2*x2 + y2*y2 + z2*z2
		angle_b = np.arccos(dot/np.sqrt(lenSq1 * lenSq2))
		angle = np.arccos(np.dot(refVector,targetVector))

		if nParams==1:
			guess += [pca_stds[i][0]]
		if nParams==2:
			guess += [pca_stds[i][0],pca_stds[i][1]]
		if nParams==3:
			guess += [pca_stds[i][0],pca_stds[i][1],angle]
		if nParams==4:
			guess += [pca_stds[i][0],pca_stds[i][1],pca_stds[i][0],pca_stds[i][1]]
		if nParams==5:
			guess += [pca_stds[i][0],pca_stds[i][1],angle,pca_stds[i][0],pca_stds[i][1]]
		thetas.append(angle)

	return guess, thetas


def superPixels(labels_, data_samples_s1, imgBlur_full, cmap, fn_id=''):
	print("Superpixels...")
	neigh = KNeighborsClassifier(n_neighbors=1)

	print(data_samples_s1.shape, labels_.shape)
	neigh.fit(data_samples_s1, labels_) 

	greyImg = np.max(imgBlur_full,axis=2)

	m1,s1 = weighted_avg_and_std(greyImg,getPoleScaleMap(greyImg.shape[1]))
	threshold = m1+(s1*nStdAboveMean)
	#threshold = np.mean(greyImg)+np.std(greyImg)*nStdAboveMean
	greyImg[greyImg<threshold] = 0
	
	w = imgBlur_full.shape[1]
	h = imgBlur_full.shape[0]
	img = np.zeros((h,w,3),dtype=np.float32)
	img_label = np.ones((h,w),dtype=np.int_)*-1
	for y in range(0,h):
		for x in range(0,w):
			intensity = greyImg[y,x]
			if intensity<threshold:
				continue # skip
			cart = xy2XYZ(x,y,w,h)
			label = neigh.predict([cart])[0]
			color = cmap(label)
			img[y,x,0] = color[0]
			img[y,x,1] = color[1]
			img[y,x,2] = color[2]
			img_label[y,x] = label
	return img_label

def samplingMethod(imgBlur,imgBlur_full,width,height,nStdAboveMean,lightIndices,nLights,nSamplesInput):
	print("-> Generating samples")
	samplingImage = np.copy(imgBlur_full)
	s1, s2, cols, s_values, nSamples = sample_spherical_biased(nSamplesInput,img=samplingImage,nStdAboveMean=nStdAboveMean,doRemoveBottom=False)
	print("n samples:",nSamples)

	outLightsImg = np.copy(imgBlur_full)
	for i in range(0,nSamples):
		clusters_pixelcoords = cartesian2xy(s1[i,0],s1[i,1],s1[i,2],width,height)
		outLightsImg[clusters_pixelcoords[1],clusters_pixelcoords[0],0] = 0
		outLightsImg[clusters_pixelcoords[1],clusters_pixelcoords[0],1] = 10000
		outLightsImg[clusters_pixelcoords[1],clusters_pixelcoords[0],2] = 0

	# Add light detected samples
	data_samples_s1 = np.copy(s1)
	data_samples_s2 = np.copy(s2)
	data_samples_values = np.copy(s_values)

	print("# Samples:",data_samples_s1.shape)
	
	# add peaks to samples
	for i in range(len(lightIndices)-1,-1,-1):
		Li = lightIndices[i]
		cart = xy2XYZ(X[Li],Y[Li],iblWidth,iblHeight)
		data_samples_s1 = np.vstack((cart,data_samples_s1))
		pixel = imgBlur_full[Y[Li],X[Li],:]
		scale = np.log1p(np.max(pixel))
		cart *= scale
		data_samples_s2 = np.vstack((cart,data_samples_s2))
		data_samples_values = np.hstack((np.max(pixel),data_samples_values))
		cols = [[0,1,0]] + cols
	print("# Samples (with lights)", data_samples_s1.shape, data_samples_s2.shape, data_samples_values.shape)

	return data_samples_s1,data_samples_values,nSamples

def clusteringMethodFunc(imgBlur_IN,imgBlur_full,clusteringMethod,nLights,nParams,nSamplesInput,nSamples,data_samples_s1):
	print("-> Associating samples to lights (clustering):",clusteringMethod)

	imgBlur = np.max(imgBlur_IN,axis=2)

	def get_cmap(n, name='rainbow'):
		return plt.cm.get_cmap(name, n)

	method_id = 1
	methods = [climbLightDistance6,climbLightDistance7]
	k = nLights

	#labels_ = graphClustering(data_samples_s2, nLights)
	if clusteringMethod=='curveClustering':
		labels_ = curveClustering(data_samples_s1, nLights, imgBlur, methods[method_id])
	elif clusteringMethod=='curveClusteringParralel':
		labels_ = curveClusteringParralel(data_samples_s1, nLights, imgBlur)
	else: # euclidean
		labels_ = euclideanClustering(data_samples_s1, nLights)

	print('unique:', np.unique(labels_).shape[0])

	cmap = get_cmap(k)

	saveClusterImg(nSamples, labels_, data_samples_s1, imgBlur_full, cmap, k=k, nSamplesInput=nSamplesInput, clusteringMethod=clusteringMethod, method_id=method_id)

	img_label = superPixels(labels_, data_samples_s1, imgBlur_full, cmap, fn_id='_'+clusteringMethod)
	#sys.exit()

	print("-> PCA")
	pca_primary_components,pca_stds = getPCA(nLights, data_samples_s1, labels_, cmap, showPlot_PCA=False)
	#sys.exit()

	print(pca_primary_components)
	print(pca_stds)
	print('nParams:',nParams)
	guess_all, thetas = getGuessFromPCA(nLights,nParams,resReduction,pca_primary_components,pca_stds)
	print(guess_all)
	print(thetas)

	return guess_all, thetas, img_label, labels_

def linear2sRGB(hdr_img, gamma=2.2, autoExposure = 1.0):
	# Autoexposure
	hdr_img = hdr_img*autoExposure

	# Brackets
	lower = hdr_img <= 0.0031308
	upper = hdr_img > 0.0031308

	# Gamma correction
	hdr_img[lower] *= 12.92
	hdr_img[upper] = 1.055 * np.power(hdr_img[upper], 1.0/gamma) - 0.055

	# HDR to LDR format
	img_8bit = np.clip(hdr_img*255, 0, 255).astype('uint8')
	return img_8bit

def addAmbience(img_IN, a):
	img = np.copy(img_IN)
	for i in range(len(a)):
		img[:, :, i] = np.maximum(img[:, :, i], a[i])
	return img



def evaluate(xyz,params, doRemoveBottom=False, isGrey=False, keepAsFlattenedArray=False):
	if isGrey:
		y = np.zeros((xyz.shape[0],xyz.shape[1]))
	else:
		y = np.zeros((xyz.shape[0],xyz.shape[1],3))
	if doRemoveBottom:
		y = removeBottom(y)
	y = y.reshape(-1)
	count = 0
	pxl_coord_all=[]
	z_all=[]
	intensity_all=[]
	for i in range(0, len(params), nParams):
		Li = lightIndices[count]
		stdx, stdy, theta = (None, None, 0.0)
		if model_id>=ModelInfo.GAUSS_SPHERE_SX[0]:
			stdx = params[i]
			stdy = params[i]
		if model_id>=ModelInfo.GAUSS_SPHERE_SX_SY[0]:
			stdy = params[i+1]
		if model_id>=ModelInfo.GAUSS_SPHERE_SX_SY_T[0]:
			theta = params[i+2]
		
		amp = 1.0
		if isGrey:
			amp = (targetImg[Y[Li],X[Li],0]+targetImg[Y[Li],X[Li],1]+targetImg[Y[Li],X[Li],2])/3
		else:
			amp = targetImg[Y[Li],X[Li],:]
		yImg = gaussianSphere(xyz, width=iblWidth, height=iblHeight, center=[X[Li],Y[Li]], amp=amp, stdx=stdx, stdy=stdy, theta=theta)
		pxl_coord = (X[Li],Y[Li]) # direction expressed as pixel coordinate
		world_coord = xy2XYZ(X[Li],Y[Li],width,height) # direction expressed as vector in xyz coordiante, the real one
		intensity = targetImg[Y[Li],X[Li]] # intensity of the light souce center
		pxl_coord_all.append(pxl_coord)
		z_all.append(world_coord)
		intensity_all.append(intensity)
		if doRemoveBottom:
			yImg = removeBottom(yImg)
		y = np.maximum(y,yImg.reshape(-1))
		count += 1
	if keepAsFlattenedArray:
		return y, pxl_coord_all, z_all, intensity_all
	else:
		if isGrey:
			return y.astype(np.float32).reshape((xyz.shape[0],xyz.shape[1])), pxl_coord_all, z_all, intensity_all
		else:
			return y.astype(np.float32).reshape((xyz.shape[0],xyz.shape[1],3)), pxl_coord_all, z_all, intensity_all

def evaluate_single(xyz,params, lightIdx, doRemoveBottom=False, isGrey=False, keepAsFlattenedArray=False):
	if isGrey:
		y = np.zeros((xyz.shape[0],xyz.shape[1]))
	else:
		y = np.zeros((xyz.shape[0],xyz.shape[1],3))
	if doRemoveBottom:
		y = removeBottom(y)
	y = y.reshape(-1)
	for i in range(0, len(params), nParams):
		Li = lightIndices[lightIdx]
		stdx, stdy, theta= (None, None, None)
		if model_id>=ModelInfo.GAUSS_SPHERE_SX[0]:
			stdx = params[i]
		if model_id>=ModelInfo.GAUSS_SPHERE_SX_SY[0]:
			stdy = params[i+1]
		if model_id>=ModelInfo.GAUSS_SPHERE_SX_SY_T[0]:
			theta = params[i+2]
		
		amp = 1.0
		if isGrey:
			amp = (targetImg[Y[Li],X[Li],0]+targetImg[Y[Li],X[Li],1]+targetImg[Y[Li],X[Li],2])/3
		else:
			amp = targetImg[Y[Li],X[Li],:]
		yImg = gaussianSphere(xyz, width=iblWidth, height=iblHeight, center=[X[Li],Y[Li]], amp=amp, stdx=stdx, stdy=stdy, theta=theta)
		if doRemoveBottom:
			yImg = removeBottom(yImg)
		y = np.maximum(y,yImg.reshape(-1))
	if keepAsFlattenedArray:
		return y
	else:
		if isGrey:
			return y.astype(np.float32).reshape((xyz.shape[0],xyz.shape[1]))
		else:
			return y.astype(np.float32).reshape((xyz.shape[0],xyz.shape[1],3))

def func_curvefit(xyz, *params):
	y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y		
def func_curvefit_single(args, *params):
	xyz = args[0]
	lightIdx = args[1]
	y= evaluate_single(xyz,params,lightIdx, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y		
def func_leastsq(params, args):
	xyz = args
	y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y-yTarget
	
def func_leastsq_single(params, arg1, arg2):
	xyz = arg1
	lightIdx = arg2
	y = evaluate_single(xyz,params,lightIdx, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y-yTarget		

if __name__ == "__main__":
	print('Spherical Gaussians IBL')
	np.random.seed(0)

	hdr_dir = "./hdr_img"
	output_dir = "./output"
	output_dir_vis = "./output_vis"
	all_files = os.listdir(hdr_dir)
	nms = [file for file in all_files if file.endswith('.exr')]
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

	if not os.path.exists(output_dir_vis):
	    os.makedirs(output_dir_vis)

	crop_tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
	tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.99)

	width = 256
	height = int(width/2)
	nStdAboveMean = 2.0
	clusteringMethod = 'curveClusteringParralel'
	nSamplesInput = width
	rough_level=0.2		# modelInfo 	= ModelInfo.GAUSS_SPHERE_SX#_SY_T#_SX2_SX3
	modelInfo 	= ModelInfo.GAUSS_SPHERE_SX_SY_T#_SX2_SX3
	model_id 	= modelInfo[0]
	nParams		= modelInfo[1]		
	print('model id:',model_id, '#params:',nParams)		
	useCurveFit = True
	useGoodGuess = False # turn this off, it is too slow and doesn't help
	runMinimisation = True
	fitAllAtOnce = True
	useAmbientValues = True
	solveAsGrey = False # True is grey based optimsation, False is colour based optimisation
	isParamOut=True			
	startTime = time.time()
	i=0
	for nm in nms[501:1001]:
		i = i + 1
		print("Processing i:",i)
		if nm.endswith('.png.exr') and ('_hdr_Ref_ibl_') in nm:
			hdr_path = hdr_dir + nm
			crop_path = hdr_dir + nm.replace("ibl","view_ldr").replace(".png.exr","_80.exr")
			print("crop_path:",crop_path)			
			h = util.PanoramaHandler()
			crop = h.read_exr(crop_path)
			crop,alpha = crop_tone(crop)  #alpha need to be multiplied back when output intensity
			alpha = alpha*3
			# crop,alpha = tone(crop)  #alpha need to be multiplied back when output intensity
			print("alpha:",alpha)

			fn 	= hdr_path
			blurRadius = np.clip(rough_level,0.0,1.0)		

			print("Loading data...")
			img	= loadData(fn, width)		

			print("Loading IBL_Rougness...",blurRadius)
			print(fn)
			# Make an instance of IBL_Rougness class and load in fn ibl as well
			iblRoughness = _rm.IBL_Roughness(fn, roughLevels=[blurRadius], nRoughLevels=None, processing_width=128, output_width=width, visible=False)
			iblRoughness.update()
			# Let's print out what we can access, including the stack of rough images
			imgBlur_full = iblRoughness.images[0,:,:,:]
			imgBlur = np.copy(imgBlur_full)


			###### Light detection and resolution reduction
			X,Y,lightIndices,nLights,threshold,ambientLightValue = processLightDetection(imgBlur,nStdAboveMean)		

			iblWidth = width #img.shape[1]
			iblHeight = int(iblWidth/2)
			img,imgBlur,X,Y,resReduction = reduceResolution(iblWidth,iblHeight,width,img,imgBlur,X,Y,lightIndices)		

			print("Fitting Spherical Gaussians...")
			# sx, sy, theta
			targetImg = np.copy(imgBlur)
			# targetImg /= np.max(targetImg)
			# im.imwrite('./output/_targetImg.exr', targetImg)
			yTarget = targetImg.reshape(-1) 		

			print("Getting guess...")		

			guess_all_basic = np.asarray(getGuess(nLights,nParams,resReduction))		

			xyz_output = getXYZData(1000,500)

			if useGoodGuess:
				#guess_all_advanced,thetas,data_samples_s1,data_samples_values, img_label, labels_ = samplingMethod(imgBlur,imgBlur_full,width,height,nStdAboveMean,lightIndices,nLights,clusteringMethod,nSamplesInput)
				data_samples_s1,data_samples_values,nSamples = samplingMethod(imgBlur,imgBlur_full,width,height,nStdAboveMean,lightIndices,nLights,nSamplesInput)
				guess_all_advanced,thetas,img_label,labels_ = clusteringMethodFunc(imgBlur,imgBlur_full,clusteringMethod,nLights,nParams,nSamplesInput,nSamples,data_samples_s1)		

			if runMinimisation==False:
				print('Complete')
				sys.exit()		

			xyz = getXYZData(iblWidth,iblHeight)
			print("Minimising...")
			if useGoodGuess:
				if fitAllAtOnce:
					############################## All (good guess)
					print('All (advanced)')
					if solveAsGrey:
						yTarget = toGrey(targetImg).reshape(-1) 
					else:
						yTarget = targetImg.reshape(-1) 		


					popt_all = []
					#try:
					if useCurveFit:
						popt_all, pcov = curve_fit(func_curvefit, xyz, yTarget, p0=guess_all_advanced) #, maxfev=10000, ftol=0.1, xtol=0.1)
					else:
						popt_all, pcov = leastsq(func_leastsq, x0=guess_all_advanced, args=(xyz)) #, maxfev=10000, ftol=0.5, xtol=0.5) 

					gaussImg, pxl_coord_all_fit, z_all_fit, intensity_all_fit = evaluate(xyz_output,popt_all,doRemoveBottom=False)
					
					if useAmbientValues:
						gaussImg = addAmbience(gaussImg, ambientLightValue)	
					gaussImg = gaussImg*alpha	
					im.imwrite(os.path.join(output_dir_vis,nm.replace(".exr","")+'_ASG.exr'), gaussImg)
					gaussImg = tone(gaussImg)[0].astype('float32') * 255.0  
					gaussImg =  Image.fromarray(gaussImg.astype('uint8'))
					gaussImg.save(os.path.join(output_dir_vis,'{}_ASG.jpg'.format(os.path.basename(hdr_path).split('.')[0])))	

			else:
				if fitAllAtOnce:
					############################## All (simple guess)
					print('All (basic)')
					if solveAsGrey:
						yTarget = toGrey(targetImg).reshape(-1) 
					else:
						yTarget = targetImg.reshape(-1) 		

					startTime = time.time()
					popt_all = []
					#try:
					if useCurveFit:
						print(xyz.shape)
						print(yTarget.shape)
						popt_all, pcov = curve_fit(func_curvefit, xyz, yTarget, p0=guess_all_basic) #, maxfev=10000, ftol=0.1, xtol=0.1)
					else:
						popt_all, pcov = leastsq(func_leastsq, x0=guess_all_basic, args=(xyz)) #, maxfev=10000, ftol=0.5, xtol=0.5) 


					gaussImg, pxl_coord_all_fit, z_all_fit, intensity_all_fit = evaluate(xyz_output,popt_all,doRemoveBottom=False)
					if useAmbientValues:
						gaussImg = addAmbience(gaussImg, ambientLightValue)
					gaussImg = gaussImg*alpha	
					im.imwrite(os.path.join(output_dir_vis,nm.replace(".exr","")+'_ASG.exr'), gaussImg)
					gaussImg = tone(gaussImg)[0].astype('float32') * 255.0  
					gaussImg =  Image.fromarray(gaussImg.astype('uint8'))
					gaussImg.save(os.path.join(output_dir_vis,'{}_ASG.jpg'.format(os.path.basename(hdr_path).split('.')[0])))			
		
			print("Output the estimated parameters...")
			data = {
			    'popt_all': np.array(popt_all), # fitting guassian stdx, stdy, theta
			    'pxl_coord_all_fit': np.array(pxl_coord_all_fit),# direction expressed as pixel coordinate
			    'z_all_fit': np.array(z_all_fit), # direction expressed as vector in xyz coordiante, the real one
			    'intensity_all_fit': np.array(intensity_all_fit)*alpha*3,# intensity of the light souce center, 3 channle will be light color
			    'ambient': np.array(ambientLightValue)*alpha*3 # constant ambient
			}		

			# Save to .npy file
			np.save(os.path.join(output_dir,nm.replace(".exr","")+'_ASG.npy'), data)		

			# Load from .npy file
			loaded_data = np.load(os.path.join(output_dir,nm.replace(".exr","")+'_ASG.npy'), allow_pickle=True).item()		

			# Access individual variables
			loaded_popt_all = loaded_data['popt_all'] # nParams (=3) * nLight
			loaded_pxl_coord_all_fit = loaded_data['pxl_coord_all_fit'] # xy (=2) * nLight
			loaded_z_all_fit = loaded_data['z_all_fit']## xyz (=3) * nLight
			loaded_intensity_all_fit = loaded_data['intensity_all_fit'] # RGB (=3) *nLight
			loaded_ambient = loaded_data['ambient']	#RGB for background	

			# Print loaded variables
			print("popt_all:", loaded_popt_all)
			print("pxl_coord_all_fit:", loaded_pxl_coord_all_fit)
			print("z_all_fit:", loaded_z_all_fit)
			print("intensity_all_fit:", loaded_intensity_all_fit)
			print("ambient:", loaded_ambient)	
	print('Complete')
