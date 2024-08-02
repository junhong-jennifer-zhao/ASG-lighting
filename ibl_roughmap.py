import os
import sys
import numpy as np
from PIL import Image
from glumpy import app, gloo, gl, data
import imageio as im

import cv2 # image resize

import argparse

class IBL_Roughness(object):
	def __init__(self, input_image_file, convolveMode=1, roughLevels=[0.1], nRoughLevels=None, processing_width=128, output_width=192, visible=False, roughLevels_y=[0.1], thetaLevels=[0.0], fresnelLevels=[1.0]):
		super(IBL_Roughness, self).__init__()
		'''
		self.convolveMode
		0: Diffuse
		1: Phong 				Roughness
		2: Blinn-Phong 			Roughness
		3: GGX 					Roughness, Fresnel
		4: Gaussian Specular	Roughness
		5: Beckmann Specular	Roughness
		6: Cook Torrance		Roughness, Fresnel
		7: Ward 				Roughness, Anisoptropic
		'''

		self.input_image_file = input_image_file
		self.visible = visible

		#
		self.maxModes = 8
		self.convolveMode = convolveMode
		if self.convolveMode >= self.maxModes:
			self.convolveMode = 1

		# Pre-provided roughLevels
		self.roughLevels = roughLevels
		self.nRoughLevels = len(self.roughLevels)

		# Other BRDF properties
		if len(roughLevels_y)<self.nRoughLevels:
			self.roughLevels_y = roughLevels
		else:
			self.roughLevels_y = roughLevels_y

		if len(thetaLevels)<self.nRoughLevels:
			self.thetaLevels = [0.0]*self.nRoughLevels
		else:
			self.thetaLevels = thetaLevels

		if len(fresnelLevels)<self.nRoughLevels:
			self.fresnelLevels = [1.0]*self.nRoughLevels
		else:
			self.fresnelLevels = fresnelLevels

		# If we provide nRoughLevels, let's generate it
		if nRoughLevels is not None:
			self.nRoughLevels = nRoughLevels
			self.roughLevels = np.linspace(1.0/self.nRoughLevels, 1, self.nRoughLevels)
			self.roughLevels_y = np.linspace(1.0/self.nRoughLevels, 1, self.nRoughLevels)
			self.thetaLevels = np.linspace(1.0/self.nRoughLevels, np.pi, self.nRoughLevels)
			self.fresnelLevels = np.linspace(1.0/self.nRoughLevels, 1, self.nRoughLevels)

		#
		self.roughLevels = np.clip(self.roughLevels,0.1,1) # range should be between 0.1 to 1.0
		self.roughLevels_y = np.clip(self.roughLevels_y,0.1,1) # range should be between 0.1 to 1.0
		self.thetaLevels = np.clip(self.thetaLevels,0.0,np.pi) # range should be between 0.0  to pi
		self.fresnelLevels = np.clip(self.fresnelLevels,0.1,1) # range should be between 0.1 to 1.0

		self.roughness = self.roughLevels[0]
		self.roughness_y = self.roughLevels_y[0]
		self.theta = self.thetaLevels[0]
		self.fresnel = self.fresnelLevels[0]
		self.images = None

		self.processing_width = processing_width
		self.processing_height = int(self.processing_width/2)

		self.output_width = output_width
		self.output_height = int(self.output_width/2)

		self.texture_np = None
		self.texture = None

		self.init()

	def update(self):
		app.run(framerate=0, framecount=1)

	def loadIBL(self, fn_new):
		if fn_new is None:
			return
		self.input_image_file = fn_new

		# Load IBL
		self.texture_np = im.imread(os.path.join(os.path.dirname(__file__), self.input_image_file))
		
		# resize it
		self.texture_np = cv2.resize(self.texture_np, (self.processing_width, self.processing_height), interpolation=cv2.INTER_CUBIC)
		self.texture_np[self.texture_np<0] = 0
		self.texture_np = np.nan_to_num(self.texture_np, nan=0.0, neginf=0.0)

		self.ibl_dtype = self.texture_np.dtype
		self.IBL_Resolution = (self.texture_np.shape[1], self.texture_np.shape[0])

	def linear2sRGB(self, hdr_img, gamma=2.2, autoExposure = 1.0):
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

	def init(self):
		self.loadIBL(self.input_image_file)

		# Make window
		self.window_main = app.Window(width=self.output_width, height=self.output_height, visible=self.visible)
		if not self.visible:
			self.window_main.hide()

		gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)


		def rotate(angle):
			self.cam_yaw = angle #(np.pi*2) * 1.0
			self.quad['u_Yaw'] = self.cam_yaw

		def saveImage(fn_id):
			outDir = './output/'
			out_name = outDir+self.input_image_file.split("/")[-1][:-4]+fn_id
			print(out_name)

			self.framebuffer.activate() 
			img = gl.glReadPixels(0, 0, self.window_main.get_size()[0], self.window_main.get_size()[1], gl.GL_RGB, gl.GL_FLOAT)
			self.framebuffer.deactivate()

			img = img.reshape((self.window_main.get_size()[1], self.window_main.get_size()[0], 3))
			img = np.flipud(img)

			if self.ibl_dtype==np.float32:
				out_name = out_name+'.exr'
				im.imwrite(out_name,img)
			else:
				out_name = out_name+'.jpg'
				result = Image.fromarray(np.uint8(img * 255))
				result.save(out_name)
			#print("Saved file as: " + out_name)

		def getImage():
			self.framebuffer.activate() 
			img = gl.glReadPixels(0, 0, self.window_main.get_size()[0], self.window_main.get_size()[1], gl.GL_RGB, gl.GL_FLOAT)
			self.framebuffer.deactivate()
			img = img.reshape((self.window_main.get_size()[1], self.window_main.get_size()[0], 3))
			img = np.flipud(img)
			return img

		def generateData_save():
			for i in range(0, self.nRoughLevels):
				self.roughness = self.roughLevels[i]
				self.roughness_y = self.roughLevels_y[i]
				self.theta = self.thetaLevels[i]
				self.fresnel = self.fresnelLevels[i]
				self.quad['u_Roughness'] = self.roughness
				self.quad['u_Roughness_y'] = self.roughness_y
				self.quad['u_Theta'] = self.theta
				self.quad['u_Fresnel'] = self.fresnel
				on_draw(0)
				saveImage("_%.2f_%.2f_%.2f_%d" % (self.roughness, self.roughness_y, self.fresnel, self.convolveMode))

		def generateData_return():
			self.images = np.zeros((self.nRoughLevels, self.window_main.height, self.window_main.width, 3), np.float32)
			for i in range(0, self.nRoughLevels):
				self.roughness = self.roughLevels[i]
				self.roughness_y = self.roughLevels_y[i]
				self.theta = self.thetaLevels[i]
				self.fresnel = self.fresnelLevels[i]
				self.quad['u_Roughness'] = self.roughness
				self.quad['u_Roughness_y'] = self.roughness_y
				self.quad['u_Theta'] = self.theta
				self.quad['u_Fresnel'] = self.fresnel
				on_draw(0)
				img = getImage()
				self.images[i, :,:,:] = img

		@self.window_main.event
		def on_init():
			self.window_main.activate()

			def read_file(filename):
				path = os.path.join(os.path.dirname(__file__), filename)
				with open(path, 'r') as file:
					return file.read()

			vertex = read_file('specMapShader.vert')
			fragment =  read_file('specMapShader.frag')

			# Flip
			self.texture = np.asarray(np.flip(np.flip(self.texture_np, axis=0), axis=1)).copy()
			if self.ibl_dtype == np.float32:
				self.texture = self.texture.view(gloo.TextureFloat2D)
			
			color_buf = np.zeros((self.window_main.height, self.window_main.width, 4), np.float32).view(gloo.TextureFloat2D)
			self.framebuffer = gloo.FrameBuffer(color=color_buf)

			self.quad = gloo.Program(vertex, fragment, count=4)
			self.quad['position'] = (-1,-1), (-1,+1), (+1,-1), (+1,+1)
			#self.quad['u_Texture'] = self.texture
			#self.quad['u_Texture'].interpolation = (gl.GL_LINEAR, gl.GL_LINEAR)
			# self.quad['u_Texture'].interpolation = (gl.GL_LINEAR_MIPMAP_LINEAR, gl.GL_LINEAR)

			gl.glClampColor(gl.GL_CLAMP_READ_COLOR, gl.GL_FALSE);
			gl.glClampColor(gl.GL_CLAMP_VERTEX_COLOR, gl.GL_FALSE);
			gl.glClampColor(gl.GL_CLAMP_FRAGMENT_COLOR, gl.GL_FALSE);

			#
			self.quad['u_Texture'] = self.texture
			self.quad['u_IBL_Resolution'] = self.IBL_Resolution
			
			self.cam_pitch = 0
			self.cam_yaw = 0
			self.warpAmount = 0.0
			self.fast = False

			self.modifier_rough_y = False
			self.modifier_theta = False
			self.modifier_fresnel = False

			self.latlong_mode = True
			self.needs_redraw = 0

			self.quad['u_Resolution'] = (self.window_main.width, self.window_main.height)
			self.quad['u_Pitch'] = self.cam_pitch
			self.quad['u_Yaw'] = self.cam_yaw
			self.quad['u_WarpAmount'] = self.warpAmount
			self.quad['u_Roughness'] = self.roughness
			self.quad['u_Roughness_y'] = self.roughness_y
			self.quad['u_Theta'] = self.theta
			self.quad['u_Fresnel'] = self.fresnel
			self.quad['u_ConvolveMode'] = self.convolveMode

			if not self.visible:
				generateData_return()

		@self.window_main.event
		def on_close():
			'The user closed the window.'
			pass

		@self.window_main.event
		def on_resize(width,height):
			self.window_main.activate()
			
			color_buf = np.zeros((height, width, 4), np.float32).view(gloo.TextureFloat2D)
			self.framebuffer = gloo.FrameBuffer(color=color_buf)

			self.quad['u_Resolution'] = (width, height)
			self.needs_redraw = 0

		@self.window_main.event
		def on_mouse_drag(x, y, dx, dy, button):
			self.window_main.activate()
			# print('Mouse drag (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f, button=%d)' % (x,y,dx,dy,button))
			if button == 2: # LMB
				s = 0.01
				self.cam_pitch = np.max([ -np.pi/2, np.min([np.pi/2, self.cam_pitch - s*dy]) ])
				self.cam_yaw = self.cam_yaw - s*dx
				self.quad['u_Pitch'] = self.cam_pitch
				self.quad['u_Yaw'] = self.cam_yaw
				self.needs_redraw = 0

		@self.window_main.event
		def on_mouse_scroll(x, y, dx, dy):
			self.window_main.activate()
			# print('Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy))

			print(self.fast, self.modifier_fresnel, self.modifier_rough_y, self.modifier_theta)

			speed = 0.05
			if self.fast:
				speed = 0.1

			#self.roughness = np.clip(self.roughness+(speed*dy), 1, 100000);
			if self.modifier_fresnel:
				self.fresnel = np.clip(self.fresnel+(speed*dy), 0.1, 1.0);
			elif self.modifier_rough_y:
				self.roughness_y = np.clip(self.roughness_y+(speed*dy), 0.1, 1.0);
			elif self.modifier_theta:
				self.theta = ((self.theta+(speed*dy)) % np.pi);
			else:
				self.roughness = np.clip(self.roughness+(speed*dy), 0.1, 1.0);
	
			print(self.roughness, self.roughness_y, self.fresnel, self.theta)

			
			self.quad['u_Roughness'] = self.roughness
			self.quad['u_Roughness_y'] = self.roughness_y
			self.quad['u_Theta'] = self.theta
			self.quad['u_Fresnel'] = self.fresnel
	
			#self.warpAmount += 0.1*dy;
			#self.quad['u_WarpAmount'] = self.warpAmount

			self.needs_redraw = 0

		@self.window_main.event
		def on_mouse_press(x, y, button):
			self.window_main.activate()
			# print('Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button))
			if button == 8: # RMB
				pass
			elif button == 4: # MMB
				pass

		@self.window_main.event
		def on_draw(dt):
			#self.window_main.activate()
			self.window_main.clear()

			# draw to frame buffer
			self.framebuffer.activate()
			self.quad.draw(gl.GL_TRIANGLE_STRIP)
			self.framebuffer.deactivate()

			# draw to main window
			self.quad.draw(gl.GL_TRIANGLE_STRIP)

		@self.window_main.event
		def on_key_press(symbol, modifiers):
			if symbol==65: # a
				self.fast = True
			if symbol==82: # r
				self.modifier_rough_y = True
			if symbol==70: # f
				self.modifier_fresnel = True
			if symbol==84: # t
				self.modifier_theta = True

			if symbol == 65289: # tab
				self.quad['u_ShowLatlong'] = self.latlong_mode = not self.latlong_mode
				self.needs_redraw = 0
			elif modifiers==2 and symbol==65: # ctrl+a
				generateData_save()
			elif modifiers==2 and symbol==83: # ctrl+s
				saveImage("_%.2f_%.2f_%.2f_%d" % (self.roughness, self.roughness_y, self.fresnel, self.convolveMode))
			elif symbol==32:
				self.convolveMode = (self.convolveMode+1) % self.maxModes
				print("Convolve mode %s" % str(self.convolveMode))
				self.quad['u_ConvolveMode'] = self.convolveMode
				self.needs_redraw = 0

			elif symbol==65361: # left
				self.convolveMode = self.convolveMode-1
				if self.convolveMode < 0:
					self.convolveMode = self.maxModes-1

				print("Convolve mode %s" % str(self.convolveMode))
				self.quad['u_ConvolveMode'] = self.convolveMode
				self.needs_redraw = 0

			elif symbol==65363: # right
				self.convolveMode = (self.convolveMode+1) % self.maxModes
				print("Convolve mode %s" % str(self.convolveMode))
				self.quad['u_ConvolveMode'] = self.convolveMode
				self.needs_redraw = 0

		@self.window_main.event
		def on_key_release(symbol, modifiers):
			if symbol==65:
				self.fast = False
			if symbol==82: # r
				self.modifier_rough_y = False
			if symbol==70: # f
				self.modifier_fresnel = False
			if symbol==84: # t
				self.modifier_theta = False

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Generates warped IBLs'
	)

	parser.add_argument(
		'ibl',
		metavar='IBL Path',
		type=str,
		default=None,
		nargs="?",
		help='Path to IBL'
	)
	parser.add_argument('--visible', dest='visible', action='store_true')
	parser.add_argument('--no-visible', dest='visible', action='store_false')
	parser.set_defaults(visible=True)

	args = parser.parse_args()
	ibl_name = args.ibl
	visible = args.visible

	if ibl_name is None:
		print('Missing args')
		sys.exit()

	v = IBL_Roughness(ibl_name, visible=visible)
	app.run()

