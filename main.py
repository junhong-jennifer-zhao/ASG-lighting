import os
import numpy as np
from util import *
from clustering import *
from geometry import *
from asg_extraction import *


def func_curvefit(xyz, *params):
	# y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, nParams, lightIndices, model_id, targetImg, X, Y, width = iblWidth, height=iblHeight, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y		
def func_curvefit_single(args, *params):
	xyz = args[0]
	lightIdx = args[1]
	y= evaluate_single(xyz,params,lightIdx, lightIndices, model_id, targetImg, X, Y,mwidth = iblWidth, height=iblHeight,  isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y

def func_leastsq(params, args):
	xyz = args
	y, pxl_coord_all, z_all, intensity_all = evaluate(xyz,params, nParams, lightIndices, model_id, targetImg, X, Y, width = iblWidth, height=iblHeight,  isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y-yTarget
	
def func_leastsq_single(params, arg1, arg2):
	xyz = arg1
	lightIdx = arg2
	y = evaluate_single(xyz,params,lightIdx, lightIndices, model_id, targetImg, X, Y,width = iblWidth, height=iblHeight, isGrey=solveAsGrey, keepAsFlattenedArray=True)
	return y-yTarget

if __name__ == "__main__":
	print('Spherical Gaussians IBL')
	np.random.seed(0)
	hdr_dir = "./hdr_img/"
	output_dir = "./output/"
	output_dir_vis = "./output_vis/"


	all_files = os.listdir(hdr_dir)
	nms = [file for file in all_files if file.endswith('.exr')]
	print("nms:",nms[1:10])
	print("nms files number:",len(nms))
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

	if not os.path.exists(output_dir_vis):
	    os.makedirs(output_dir_vis)

	crop_tone = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
	tone = TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.99)

	width = 256
	height = int(width/2)
	nStdAboveMean = 2.0
	#clusteringMethod = 'curveClustering'
	clusteringMethod = 'curveClusteringParralel'
	#clusteringMethod = 'euclidean'
	nSamplesInput = width
	rough_level=0.2		# modelInfo 	= ModelInfo.GAUSS_SPHERE_SX#_SY_T#_SX2_SX3
	modelInfo 	= ModelInfo.GAUSS_SPHERE_SX_SY_T#_SX2_SX3
	model_id 	= modelInfo[0]
	nParams		= modelInfo[1]	
	print("nParams:", nParams)	
	print('model id:',model_id, '#params:',nParams)		
	useCurveFit = True
	useGoodGuess = False 
	runMinimisation = True
	fitAllAtOnce = True
	useAmbientValues = True
	solveAsGrey = False # True is grey based optimsation, False is colour based optimisation
	isParamOut=True			
	startTime = time.time()
	i=0
	for nm in nms[:]: #429
		i = i + 1
		print("Processing i:",i)
		if nm.endswith('.exr'):
			hdr_path = hdr_dir + nm
			alpha = 1
			# crop,alpha = tone(crop)  #alpha need to be multiplied back when output intensity
			print("alpha:",alpha)

			fn 	= hdr_path
			blurRadius = np.clip(rough_level,0.0,1.0)		

			print("Loading data...")
			img	= loadData(fn, width)		

			print("Loading IBL_Rougness...",blurRadius)
			print(fn)

			blurAmount =5
			imgBlur_full = cv2.GaussianBlur(img, (blurAmount, blurAmount), 0)
			imgBlur = np.copy(imgBlur_full)


			###### Light detection and resolution reduction
			X,Y,lightIndices,nLights,threshold,ambientLightValue = processLightDetection(imgBlur,nStdAboveMean)		

			iblWidth = width #img.shape[1]
			iblHeight = int(iblWidth/2)
			img,imgBlur,X,Y,resReduction = reduceResolution(iblWidth,iblHeight,width,img,imgBlur,X,Y,lightIndices)		

			print("Fitting Spherical Gaussians...")
			# sx, sy, theta
			targetImg = np.copy(imgBlur)
			yTarget = targetImg.reshape(-1) 		

			print("Getting guess...")		

			guess_all_basic = np.asarray(getGuess(nLights,nParams,resReduction))		

			xyz_output = getXYZData(1000,500)

			if useGoodGuess:
				#guess_all_advanced,thetas,data_samples_s1,data_samples_values, img_label, labels_ = samplingMethod(imgBlur,imgBlur_full,width,height,nStdAboveMean,lightIndices,nLights,clusteringMethod,nSamplesInput)
				data_samples_s1,data_samples_values,nSamples = samplingMethod(imgBlur,imgBlur_full,width,height,nStdAboveMean,lightIndices,nLights,nSamplesInput,X,Y)
				guess_all_advanced,thetas,img_label,labels_ = clusteringMethodFunc(imgBlur,imgBlur_full,clusteringMethod,nLights,nParams,nSamplesInput,nSamples,data_samples_s1, width, height,nStdAboveMean,X,resReduction)		

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
					if useCurveFit:
						popt_all, pcov = curve_fit(func_curvefit, xyz, yTarget, p0=guess_all_advanced) #, maxfev=10000, ftol=0.1, xtol=0.1)
					else:
						popt_all, pcov = leastsq(func_leastsq, x0=guess_all_advanced, args=(xyz)) #, maxfev=10000, ftol=0.5, xtol=0.5) 

					gaussImg, pxl_coord_all_fit, z_all_fit, intensity_all_fit = evaluate(xyz_output,popt_all, nParams, lightIndices, model_id, targetImg, X, Y, width = iblWidth, height=iblHeight, isGrey=solveAsGrey, doRemoveBottom=False)


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
					if useCurveFit:
						print(xyz.shape)
						print(yTarget.shape)
						popt_all, pcov = curve_fit(func_curvefit, xyz, yTarget, p0=guess_all_basic) #, maxfev=10000, ftol=0.1, xtol=0.1)
					else:
						popt_all, pcov = leastsq(func_leastsq, x0=guess_all_basic, args=(xyz)) #, maxfev=10000, ftol=0.5, xtol=0.5) 


					gaussImg, pxl_coord_all_fit, z_all_fit, intensity_all_fit = evaluate(xyz_output,popt_all, nParams, lightIndices, model_id, targetImg, X, Y, width = iblWidth, height=iblHeight, isGrey=solveAsGrey, doRemoveBottom=False)
					                                                           



					if useAmbientValues:
						gaussImg = addAmbience(gaussImg, ambientLightValue)
					gaussImg = gaussImg*alpha*10	
					im.imwrite(os.path.join(output_dir_vis,nm.replace(".exr","")+'_ASG.exr'), gaussImg)
					gaussImg = tone(gaussImg)[0].astype('float32') * 255.0  
					gaussImg =  Image.fromarray(gaussImg.astype('uint8'))
					gaussImg.save(os.path.join(output_dir_vis,'{}_ASG.jpg'.format(os.path.basename(hdr_path).split('.')[0])))			
		
			print("Output the estimated parameters...")
			data = {
			    'popt_all': np.array(popt_all), # fitting guassian stdx, stdy, theta
			    'pxl_coord_all_fit': np.array(pxl_coord_all_fit),# direction expressed as pixel coordinate
			    'z_all_fit': np.array(z_all_fit), # direction expressed as vector in xyz coordiante, the real one
			    'intensity_all_fit': np.array(intensity_all_fit)*alpha*30,# intensity of the light souce center, 3 channle will be light color
			    'ambient': np.array(ambientLightValue)*alpha*30 # constant ambient
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

	# time_all = time.time()-startTime
	# print('Time (all):',time_all)			

	print('Complete')
