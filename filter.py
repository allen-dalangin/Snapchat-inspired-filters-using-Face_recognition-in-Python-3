#####Python Photobooth by Allen Joshua Dalangin#####

#import all necessary modules
import face_recognition
import numpy as np
import cv2

#set variables for capturing image and the filter art to be used
mask = cv2.imread('rudolph_filter.png',-1) 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

#set class for face object
class Face:

	def __init__(self, image):

		self.image = image

#set class for mask object
class Mask: 

	def __init__(self, mask):

		self.mask = mask

#set class for mask operation functions
class Mask_ops:

	#function used to overlay an image onto the incoming video 	stream
	def overlay(self, frame, mask, pos=(0,0), scale = 1): 

		#dimentions of the mask are first extracted, to be used as a basis for overlay
		mask = cv2.resize(mask, (0,0), fx=scale, fy=scale)
		h, w, _ = mask.shape
		rows, cols, _ = frame.shape
		y, x = pos[0], pos[1]

		#each pixel is first distinguised between opaque and transparent. opaque pixels are chosen to be overlaid on the video stream
		for i in range(h):
			for j in range(w):
				if x + i >= rows or y + j >= cols:
					continue

				alpha = float(mask[i][j][3] / 255.0)
				frame[x + i][y + j] = alpha * mask[i][j][:3] + (1 - alpha) * frame[x + i][y + j]

		return frame

#set class for functions to be used on the main program
class Main:

	#function for displaying the instructions on how to use the program
	def instruct(self):

		print("""

		****Welcome to the Python Photobooth****

		See your face be transformed with cute masks!
		
		To save photos, press SPACE
		To end program, press ESC

		""")

	#function for capturing the image and saving to home folder
	def capture_img(self, image, count):

		img_name = "Photobooth_{}.png".format(count)
		cv2.imwrite(img_name, image)

		print("{} captured!".format(img_name))

#define classes to be used in the program			
mask_ops = Mask_ops()
main = Main()

main.instruct()
count = 0

while True:

#image is retrieved frame by frame
	ret, image = cap.read()

#the coordinates of the face is located in the frame
	image_frame = image[:, :, ::-1]
	extracted = face_recognition.face_locations(image_frame)
	faces = [(0,0,0,0)]

	if extracted != []:

		faces = [[extracted[0][3], extracted[0][0], abs(extracted[0][3] - extracted[0][1]) + 100, abs(extracted[0][0] - extracted[0][2])]]

	#face borders are adjusted to calibrate the filter
		for (x, y, w, h) in faces:

			x -= 20
			w -= 60
			y -= 40
			h += 20

	#measurements are extracted in order to position the mask			

			mask_symin = int(y - 3 * h / 5)
			mask_symax = int(y + 8 * h / 5)

			sh_mask = mask_symax - mask_symin

			mask_sxmin = int(x - 3 * w / 5)
			mask_sxmax = int(x * w / 5)

			sw_mask = mask_sxmax - mask_sxmin

	#mask is resized based on the coordinates of the face border in order for the filter to match

			face_frame = image[mask_symin:mask_symax, x:x+w]
			mask_resized= cv2.resize(mask, (w, sh_mask),interpolation=cv2.INTER_CUBIC)

	#mask image is overlaid on the video stream
			mask_ops.overlay(face_frame, mask_resized)

	#video stream is viewed live
	cv2.imshow('Python Photobooth', image)

	k = cv2.waitKey(1)

	if k%256 == 32:

	#if SPACE key is pressed, an screengrab will be saved in the program folder
		main.capture_img(image, count)
		count += 1

	#if ESC key is pressed, the program will end
	if k%256 == 27:

		break

#video capture is stopped and all windows are closed
cap.release()
cv2.destroyAllWindows()

	

