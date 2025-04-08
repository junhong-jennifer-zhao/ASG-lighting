import numpy as np
from geometry import *
from util import *

from scipy.stats import rv_discrete
import time

from multiprocessing import Pool
from itertools import repeat

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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
	#Cluster sampling points into groups based on spherical geometry or distances.
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

def saveClusterImg(nSamples, labels_, data_samples_s1, imgBlur_full, cmap,width,height, nLights, k='', nSamplesInput='', clusteringMethod='', method_id=-1):
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

def getPCA(nLights, data_samples_s1, labels_, X, width, cmap, n_components = 2, showPlot_PCA = False):
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


def superPixels(labels_, data_samples_s1, imgBlur_full, nStdAboveMean, cmap, fn_id=''):
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

def samplingMethod(imgBlur,imgBlur_full,width,height,nStdAboveMean,lightIndices,nLights,nSamplesInput,X,Y):
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
		cart = xy2XYZ(X[Li],Y[Li],width,height)
		data_samples_s1 = np.vstack((cart,data_samples_s1))
		pixel = imgBlur_full[Y[Li],X[Li],:]
		scale = np.log1p(np.max(pixel))
		cart *= scale
		data_samples_s2 = np.vstack((cart,data_samples_s2))
		data_samples_values = np.hstack((np.max(pixel),data_samples_values))
		cols = [[0,1,0]] + cols
	print("# Samples (with lights)", data_samples_s1.shape, data_samples_s2.shape, data_samples_values.shape)

	return data_samples_s1,data_samples_values,nSamples

def clusteringMethodFunc(imgBlur_IN,imgBlur_full,clusteringMethod,nLights,nParams,nSamplesInput,nSamples,data_samples_s1,width,height,nStdAboveMean,X,resReduction):
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

	saveClusterImg(nSamples, labels_, data_samples_s1, imgBlur_full, cmap, width, height,nLights, k=k, nSamplesInput=nSamplesInput, clusteringMethod=clusteringMethod, method_id=method_id)

	img_label = superPixels(labels_, data_samples_s1, imgBlur_full,nStdAboveMean, cmap, fn_id='_'+clusteringMethod)
	#sys.exit()

	print("-> PCA")
	pca_primary_components,pca_stds = getPCA(nLights, data_samples_s1, labels_,X, width, cmap, showPlot_PCA=False)
	#sys.exit()

	print(pca_primary_components)
	print(pca_stds)
	print('nParams:',nParams)
	guess_all, thetas = getGuessFromPCA(nLights,nParams,resReduction,pca_primary_components,pca_stds)
	print(guess_all)
	print(thetas)

	return guess_all, thetas, img_label, labels_


