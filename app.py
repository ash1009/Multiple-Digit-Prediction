import pickle
import cv2
import numpy as np
from distutils.log import debug 
from fileinput import filename 
from flask import *

app = Flask(__name__)

#Fucntion to crop the sorted images and modify it to fit the prediction model and predict the digit
def predict_digit(x, y, w, h, thresh):
	# Cropping out the digit from the image corresponding to the current contours in the for loop
	digit = thresh[y:y+h, x:x+w]
	
	# Resizing that digit to (18, 18)
	resized_digit = cv2.resize(digit, (18,18), interpolation = cv2.INTER_AREA)
	
	# Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
	padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

	# Create the sharpening kernel 
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

	# Sharpen the image 
	sharpened_image = cv2.filter2D(padded_digit, -1, kernel) 

	model = pickle.load(open('digit_predict.pkl', 'rb'))
	
	# Predicting the digit
	prediction = np.argmax(model.predict(sharpened_image.reshape(1, 28, 28, 1)))
	return prediction

def make_prediction(image):
	grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(grey.copy(), 100, 255, cv2.THRESH_BINARY_INV) 
	contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)                

	actual_contours_bound = []

	for c in contours:
		area = cv2.contourArea(c)
		if area > 50: #Strict threshold... can change
			x,y,w,h = cv2.boundingRect(c)
			
			#Increasing boundary of rectangle
			#Values can be changed if/when necessary
			x -= 5
			y -= 5
			w += 10
			h += 10
			
			# Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
			cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)

			actual_contours_bound.append((x, y, w, h))  
	
	#Sorting the contours of the numbers from left-to-right and top-to-bottom
	actual_contours_arr = np.array(actual_contours_bound)

	# Calculate maximum rectangle height
	max_height = np.max(actual_contours_arr[::, 3])

	# Sort the contours by y-value
	by_y = sorted(actual_contours_bound, key=lambda x: x[1])  #y values

	line_y = by_y[0][1]       #first y
	line = 1
	by_line = []

	# Assign a line number to each contour
	for x, y, w, h in by_y:
		if y > line_y + max_height:
			line_y = y
			line += 1
		by_line.append((line, x, y, w, h))

	# This will now sort automatically by line then by x
	contours_sorted = [(x, y, w, h) for line, x, y, w, h in sorted(by_line)]

	#Code block to predict the numbers in the correct order and print them accordingly
	first_y = contours_sorted[0][1]
	last_x, last_y, last_w, last_h = contours_sorted[-1]
	number_predict = []
	all_predicted_digit = []

	for i in contours_sorted:
		x, y, w, h = i

		if abs(y-first_y) <= 90: #Strict theshold of pixels for image to be considered to be on the same line, can be changed
			prediction = predict_digit(x, y, w, h, thresh)
			number_predict.append(prediction)

			if x == last_x and y == last_y and w == last_w and h == last_h:
				final_number = int(''.join([str(x) for x in number_predict]))
				print('Predicted Number: {}'.format(final_number))
				all_predicted_digit.append(final_number)
				return all_predicted_digit
		else:
			final_number = int(''.join([str(x) for x in number_predict]))
			print('Predicted Number: {}'.format(final_number))
			all_predicted_digit.append(final_number)

			first_y = y
			number_predict.clear()
			prediction = predict_digit(x, y, w, h, thresh)
			number_predict.append(prediction)


@app.route('/') 
def main(): 
	return render_template("Index.html") 

@app.route('/test') 
def test(): 
	return render_template("Index.html") 

import os

@app.route('/success', methods=['POST']) 
def success(): 
    if request.method == 'POST': 
        f = request.files['file']
        image_name = f.filename.lower()
        valid_extensions = ['.jpg', '.jpeg', '.png']

        if any(image_name.endswith(ext) for ext in valid_extensions):
            save_path = os.path.join('static/images', image_name)
            f.save(save_path)
            picture = cv2.imread(save_path)

            predictions = make_prediction(picture)
            return render_template("Prediction.html", name=predictions)
        else:
            return {"Error": "Select a valid image file."}

@app.route('/back', methods = ['POST']) 
def back(): 
	if request.method == 'POST': 
		return render_template("Index.html") 

if __name__ == '__main__': 
	app.run("0.0.0.0", debug=True)