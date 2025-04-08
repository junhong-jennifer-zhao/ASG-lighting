import numpy as np


def sphericalToXY(v, l, denomenator):
	# Converts spherical coordinates to a 2D XY coordinate system
	return (v - l) * denomenator

def lat2y(lat, height):
	# Transform latitude values into pixel-based Y and X coordinates respectively for environmental maps
	latDen 	= (height-1)/np.pi
	y 	= sphericalToXY(np.pi/2, lat, latDen)
	if np.isscalar(y):
		return int(round(y))
	else:
		return np.round(y).astype(int)

def lon2x(lon, width):
	#Transform longitude values into pixel-based Y and X coordinates respectively for environmental maps
	longDen = (width-1)/(np.pi*2);
	x 	= sphericalToXY(np.pi, lon, longDen)
	if np.isscalar(x):
		return int(round(x))
	else:
		return np.round(x).astype(int)

def ll2xy(lat,lon,width,height):
	# Combines latitude and longitude into XY coordinate values.
	return lon2x(lon,width), lat2y(lat,height)

def cartesian2latLon(x, y, z):
	# Converts Cartesian coordinates (XYZ) to geographic latitude/longitude
	lat = np.arcsin(y)
	lon = np.arccos(np.clip(z/np.cos(lat),-1,1)) * np.sign(x)
	return lat,lon

def cartesian2xy(x,y,z,w,h):
	# Converts Cartesian coordinates (XYZ) to pixel-based image coordinates
	lat,lon = cartesian2latLon(x,y,z)
	x,y = ll2xy(lat,lon,w,h)
	return x,y
	
def latLon2Cartesian(lat, lon, r=1):
	# Converts latitude/longitude to 3D Cartesian coordinates on a unit sphere
	x = r * np.sin(lon) * np.cos(lat)
	y = r * np.sin(lat)
	z = r * np.cos(lon) * np.cos(lat)

	if not np.isscalar(x):
		y = np.repeat(y, x.shape[1], axis=1)

	return np.asarray([x,y,z])

def xyToSpherical(v, numerator, denomenator):
	# Perform reverse transformations from pixel XY coordinates back to spherical coordinates
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
	# Computes the normalized cross-product of two vectors.
	vec = np.cross(x,y)
	return vec/np.linalg.norm(vec)


def getXYZData(w,h):
	#Generates XYZ Cartesian data grids for spherical environments.
	x = np.arange(0,w)
	y = np.arange(0,h).reshape(h,1)
	latLon = xy2ll(x,y,w,h)
	cart = xy2XYZ(x,y,w,h)
	return np.swapaxes(np.array(list(zip(cart[0],cart[1],cart[2]))),1,2)
