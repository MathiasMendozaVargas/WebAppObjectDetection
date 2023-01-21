##############################
# OBJECT DETECTION WEB APP
# BY: MATHIAS MENDOZA / 2023
# pov: don't judge the UI hehe
###############################


# Import the libraries and tensorflow models
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, render_template ,send_from_directory, abort
from flask_assets import Environment, Bundle
import os

# customize your API through the following parameters
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3-tiny.tf'
tiny = True                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 80                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# Initialize Flask application
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

assets = Environment(app)
assets.url = app.static_url_path

# Being Sassy
scss = Bundle('base.scss', 'layout.scss', filters='pyscss', output='all.css')
assets.register('scss_all', scss)


# API that returns JSON with classes found in images
@app.route('/', methods=['GET'])
def get_Home():
    return render_template('index.html')

@app.route('/results')
def displayResults():
    return render_template('results.html')

# Handle page routing with Flask, and GET AND POST REQUEST
# API that returns image with detections on it
@app.route('/', methods= ['POST'])
def get_image():
    image = request.files["file"]
    image_name = image.filename
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
    img_raw = tf.image.decode_image(
        open(os.path.join(app.config['UPLOAD_FOLDER'], image_name), 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(app.config['RESULTS_FOLDER'] + 'detection.jpg', img)
    print('output saved to: {}'.format(app.config['RESULTS_FOLDER'] + 'detection.jpg'))
    
    # prepare image for response
    # _, img_encoded = cv2.imencode('.png', img)
    # response = img_encoded.tostring()
    final_result = output_path + 'detection.jpg'
    
    #remove temporary image
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

    try:
        return displayResults()
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)