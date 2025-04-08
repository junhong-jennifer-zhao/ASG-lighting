import numpy as np
import OpenEXR
import Imath
import cv2
from scipy import interpolate
import imageio as im

def removeBottom(img):
	#Excludes the bottom hemisphere from images, assuming symmetrical environmental maps.
	#rint(img.shape)
	height = img.shape[0]
	if len(img.shape)==3:
		return img[:int(height/2),:,:]
	return img[:int(height/2),:]

def loadData(fn, width):
	# Loads and resizes a high-dynamic-range (HDR) image to a specified width.
	img = im.imread(fn)
	height = int(width/2)
	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	return img


def getWeights(iblWidth):
	# Generate latitude-based weight maps for balancing equirectangular distortions
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

def toGrey(x):
	if len(x.shape)>2:
		return (x[:,:,0]+x[:,:,1]+x[:,:,2])
	return x


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



def addAmbience(img_IN, a):
	img = np.copy(img_IN)
	for i in range(len(a)):
		img[:, :, i] = np.maximum(img[:, :, i], a[i])
	return img


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


class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img.astype('float32'),alpha


class PanoramaHandler(object):
    def __init__(self):
        super(PanoramaHandler, self).__init__()

    @staticmethod
    def rgb_to_intenisty(rgbs):
        intensity = 0.2126 * rgbs[..., 0] + 0.7152 * rgbs[..., 1] + 0.0722 * rgbs[..., 0]
        return intensity

    @staticmethod
    def read_exr(exr_path):
        File = OpenEXR.InputFile(exr_path)
        PixType = Imath.PixelType(Imath.PixelType.FLOAT)
        DW = File.header()['dataWindow']
        Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
        rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
        r = np.reshape(rgb[0], (Size[1], Size[0]))
        g = np.reshape(rgb[1], (Size[1], Size[0]))
        b = np.reshape(rgb[2], (Size[1], Size[0]))
        hdr = np.zeros((Size[1], Size[0], 3), dtype=np.float32)
        hdr[:, :, 0] = r
        hdr[:, :, 1] = g
        hdr[:, :, 2] = b
        return hdr

    @staticmethod
    def read_hdr(hdr_path):
        hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        hdr_img = hdr_img[..., ::-1]
        return hdr_img

    @staticmethod
    def horizontal_rotate_panorama(hdr_img, deg):
        shift = int(deg / 360.0 * hdr_img.shape[1])
        out_img = np.roll(hdr_img, shift=shift, axis=1)
        return out_img

    @staticmethod
    def generate_steradian(height, width, multiply=True):
        steradian = np.linspace(0, height, num=height, endpoint=False) + 0.5
        steradian = np.sin(steradian / height * np.pi)
        steradian = np.tile(steradian.transpose(), (width, 1))
        steradian = steradian.transpose()
        if multiply:
            pixel_area = ((2 * np.pi) / width) * ((1 * np.pi) / height)
            steradian *= pixel_area
        return steradian.astype(np.float32)

    @staticmethod
    def prepare_gt_panorama(hdr_img, threshold=None):
        weight = PanoramaHandler.generate_steradian(height=hdr_img.shape[0], width=hdr_img.shape[1])
        hdr_intensity = PanoramaHandler.rgb_to_intenisty(hdr_img)

        if threshold is None or threshold < 0.0:
            threshold = hdr_intensity.max() / 20.

        mask = np.where(hdr_intensity < threshold)

        if mask[0].size != 0:
            ambient = np.sum(hdr_img[mask] * weight[mask][:, np.newaxis], axis=0, dtype=np.float32) / np.sum(
                weight[mask],
                dtype=np.float32)
        else:
            ambient = np.zeros([3], dtype=np.float32)

        hdr_img[mask] = 0.0
        return hdr_img, ambient

    @staticmethod
    def resize_panorama(hdr_img, new_shape):
        if isinstance(new_shape, tuple) and len(new_shape) == 2:
            hdr_img = cv2.resize(hdr_img, new_shape, interpolation=cv2.INTER_AREA)
        elif isinstance(new_shape, int):
            hdr_img = cv2.resize(hdr_img, (2 * new_shape, new_shape), interpolation=cv2.INTER_AREA)
        return hdr_img

    @staticmethod
    def crop_panorama(hdr_img, fov_deg, crop_image_h=720, crop_image_aspect_ratio="4:3"):
        # img must be float
        if hdr_img.dtype == np.uint8:
            hdr_img = hdr_img / 255.0

        numerator, denominator = [int(x) for x in crop_image_aspect_ratio.split(":")]
        ratio = numerator / denominator
        crop_image_w = int(crop_image_h * ratio)

        # print(crop_image_w, crop_image_h)
        scl = np.tan(np.deg2rad(fov_deg) / 2)
        sample_x, sample_y = np.meshgrid(
            np.linspace(-scl, scl, crop_image_w),
            np.linspace(-scl / ratio, scl / ratio, crop_image_h)
        )

        r = np.sqrt(sample_y * sample_y + sample_x * sample_x + 1)
        sample_x /= r
        sample_y /= r
        sample_z = np.sqrt(1 - sample_y * sample_y - sample_x * sample_x)
        # convert to polar
        azimuth = np.arctan2(sample_x, sample_z)  # [-Pi, Pi]
        elevation = np.arcsin(sample_y)  # [-Pi/2, Pi/2]
        # print(azimuth, elevation)

        # normalize to [0, 1]
        x = (1 + azimuth / np.pi) / 2
        y = (1 + elevation / (np.pi / 2)) / 2

        img_h = hdr_img.shape[0]
        img_w = hdr_img.shape[1]
        x = x * img_w
        y = y * img_h

        my_interpolating_function = interpolate.RegularGridInterpolator(
            (np.arange(0, hdr_img.shape[0]), np.arange(0, hdr_img.shape[1])), hdr_img)
        points = np.c_[y.ravel(), x.ravel()]
        out = my_interpolating_function(points).reshape((x.shape[0], x.shape[1], -1))
        return out

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

def tonemapping(im):

    power_im = np.power(im, 1 / 1.4)
    # print (np.amax(power_im))
    non_zero = power_im > 0
    if non_zero.any():
        r_percentile = np.percentile(power_im[non_zero], 99.9)
    else:
        r_percentile = np.percentile(power_im, 90)
    alpha = 0.99 / (r_percentile + 1e-10)
    tonemapped_im = np.multiply(alpha, power_im)

    tonemapped_im = np.clip(tonemapped_im, 0, 1)
    return tonemapped_im
    # hdr = tonemapped_im * 255.0
    # hdr = Image.fromarray(hdr.astype('uint8'))
    # hdr.save(sv_path)

def makeBinaryMaskFromIndices(shape, coords):
    binaryMask = np.zeros(shape, dtype=int)
    for coord in coords:
        y, x = coord
        y = int(y)
        x = int(x)
        binaryMask[y,x] = 1
    return binaryMask



def normalize(x):
    if np.dot(x,x)==0:
        return x
    return x/np.linalg.norm(x)


def cartesian_to_polar(xyz):
    # theta = np.arccos(np.clip(xyz[2], -1.0, 1.0))
    theta = np.arccos(np.clip(xyz[:,2], -1.0, 1.0))
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    # return phi, theta
    return np.stack((phi, theta), axis=1)   

def polar_to_cartesian(phi_theta):
    x = np.sin(phi_theta[:, 1]) * np.cos(phi_theta[:, 0])
    y = np.sin(phi_theta[:, 1]) * np.sin(phi_theta[:, 0])
    z = np.cos(phi_theta[:, 1])
    return np.stack((x, y, z), axis=1)

def sphere_points(n=128):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    # xyz = points
    # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return points

def resize_panorama(hdr_img, new_shape):
    if isinstance(new_shape, tuple) and len(new_shape) == 2:
        hdr_img = cv2.resize(hdr_img, new_shape, interpolation=cv2.INTER_AREA)
    elif isinstance(new_shape, int):
        hdr_img = cv2.resize(hdr_img, (2 * new_shape, new_shape), interpolation=cv2.INTER_AREA)
        return hdr_img

