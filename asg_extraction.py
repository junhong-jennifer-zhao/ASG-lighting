import os
import numpy as np
import time
from PIL import Image
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

import cv2
from util import *
from geometry import *

class ModelInfo():
	# ID, nParams
	GAUSS_SPHERE_SX 				= [0,1]
	GAUSS_SPHERE_SX_SY 				= [1,2]
	GAUSS_SPHERE_SX_SY_T 			= [2,3]


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

def detectAllLights(imgInput, max_sigma=30, threshold=None, nStdAboveMean=0.0, blurAmount=5, upperHemisphereOnly=False):
	#Detects light sources in an image by identifying local maxima and thresholding.

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
	#Combines multiple steps for detecting light coordinates and threshold calculations.
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





def evaluate(xyz,params, nParams, lightIndices, model_id, targetImg, X, Y, width, height, doRemoveBottom=False, isGrey=False, keepAsFlattenedArray=False):
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
		yImg = gaussianSphere(xyz, width=width, height=height, center=[X[Li],Y[Li]], amp=amp, stdx=stdx, stdy=stdy, theta=theta)
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

def evaluate_single(xyz,params, lightIdx, lightIndices, model_id, targetImg, X, Y, width, height, doRemoveBottom=False, isGrey=False, keepAsFlattenedArray=False):
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
		yImg = gaussianSphere(xyz, width=width, height=height, center=[X[Li],Y[Li]], amp=amp, stdx=stdx, stdy=stdy, theta=theta)
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



# def func_curvefit(xyz, *params):
# 	nParams, lightIndices, isGrey=params
# 	y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, nParams, lightIndices, isGrey, keepAsFlattenedArray=True)
# 	return y		
# def func_curvefit_single(xyz, *params,args):
# 	lightIdx = args[0]
# 	lightIndices = args[1]
# 	isGrey = [args[2]]
# 	y= evaluate_single(xyz,params,lightIdx, lightIndices, isGrey, keepAsFlattenedArray=True)
# 	return y		
# def func_leastsq(params, args):
# 	xyz = args[0]
# 	nParams= args[1] 
# 	lightIndices= args[2]
# 	isGrey= args[3]
# 	y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, nParams, lightIndices, isGrey, keepAsFlattenedArray=True)
# 	return y-yTarget
	
# def func_leastsq_single(params, args):
# 	xyz = args[0]
# 	lightIdx = args[1]
# 	lightIndices = args[2]
# 	isGrey = args[3]
# 	y = evaluate_single(xyz,params,lightIdx, lightIndices, isGrey, keepAsFlattenedArray=True)
# 	return y-yTarget	

	
