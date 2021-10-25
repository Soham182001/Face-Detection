from flask import Flask, render_template, request, Response, flash
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from threading import Thread



app = Flask(__name__)



global capture, switch, camera
capture = 0
switch = 0



# Machine Learning Part Starts

# function to get the output layer names 
# in the architecture
def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers



# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    label = str(classes[class_id])

    color = COLORS[0]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 4)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def YoloV3(photo):
    image = photo

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # read class names from text file
    classes = None
    with open('classes1.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes 
    COLORS = [[0,255,77,0.5]]
    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolov3_training_1000.weights', 'yolov3_testing.cfg')

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
#     print(blob)

    # set input blob for the network
    net.setInput(blob)
    
    # run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer get the confidence, class id, bounding box params and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                print("Face detected")
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    print(boxes)
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining after nms and draw bounding box
    faces = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        faces = faces+1
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

    # display output image cv2.imshow("object detection", image)
    return image, faces

# Machine Learning Part Ends



def get_face():
	global camera
	global capture
	while switch:
		success, frame = camera.read()
		if not success:
			pass
		else:
			if(capture):
				capture = 0
				cv2.imwrite('static/img.png', frame)
			try:
				ret, buffer = cv2.imencode('.jpg',frame)
				frame = buffer.tobytes()
				yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
			except Exception as e:
				pass



@app.route('/requests',methods = ['POST','GET'])
def tasks():
	global switch, camera
	if request.method == 'POST':
		if request.form.get('on_off') == 'On/Off':
			if switch == 1:
				switch = 0
				camera.release()
				cv2.destroyAllWindows()
			else:
				camera = cv2.VideoCapture(0)
				switch = 1
		elif request.form.get('take_photo') == 'Take_Photo':
			global capture
			capture = 1
	elif request.method == 'GET':
		return render_template("take_photo.html")
	return render_template("take_photo.html")



@app.route('/video_feed')
def video_feed():
	return Response(get_face(), mimetype = 'multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
	return render_template("index.html", data = "Hey")


# Prediction From Choosed Photo

@app.route('/predictionChoosedPhoto', methods = ["POST"])
def predictionChoosedPhoto():

	img = request.files['img']

	img.save("static/img.png")
	
	image1, faces_in_image = YoloV3(cv2.imread("static/img.png"))
	cv2.imwrite("static/Prediction.png",image1)
	print(faces_in_image)
	return render_template("prediction.html", num_of_face = faces_in_image)



# Prediction From Taken Photo

@app.route('/predictionTakenPhoto', methods=["POST"])
def predictionTakenPhoto():
	global switch
	image1, faces_in_image = YoloV3(cv2.imread("static/img.png"))
	if switch == 1:
		switch = 0
		camera.release()
		cv2.destroyAllWindows()
	cv2.imwrite("static/Prediction.png",image1)
	print(faces_in_image)
	return render_template("prediction.html", num_of_face = faces_in_image)



@app.route('/cleardata', methods=["POST"])
def cleardata():
	if os.path.exists("static/img.png"):
		os.remove("static/img.png")
	if os.path.exists("static/Prediction.png"):
		os.remove("static/Prediction.png")
	return render_template("index.html")


@app.route('/select', methods = ["POST"])
def get_ahead():
	if request.form.get('type') == 'LocalMachine':
		return render_template("get_photo.html")
	elif request.form.get('type') == 'TakePhoto':
		return render_template("take_photo.html")
	else:
		return render_template("index.html")


if __name__ == "__main__":
	app.run(debug=True)










