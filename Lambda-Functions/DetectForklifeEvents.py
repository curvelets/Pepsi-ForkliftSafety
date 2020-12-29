#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
""" A sample lambda for object detection"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import boto3
import math
from datetime import datetime
from pyzbar import pyzbar
import mo

# Create an IoT client for sending to messages to the cloud.
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
pep_topic = 'dt/forklifts/XXXX/camera' #'dt/forklifts/forklift 1/camera'
lastForkliftAssetId = 'forklift 1'

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()
        
def processCarsBatch(aCars, nCars):
    
    if(len(aCars) == 0):
        print('No cars in this batch')
    si = nCars -1
    for i in range(nCars):
        if(len(aCars[i]) != 0):
            si = i
            break
    ei = 0
    for i in range(nCars):
        if(len(aCars[nCars-1-i]) != 0):
            ei = nCars-1-i
            break
    print(aCars,si, ei)    
    if (si == nCars-1 or ei == 0 or ei-si < 1):
        print('Not enough data in this batch')
        return
    sPosx = (aCars[si]['xmax'] + aCars[si]['xmin']) / 2
    sPosy = (aCars[si]['ymax'] + aCars[si]['ymin']) / 2
    ePosx = (aCars[ei]['xmax'] + aCars[ei]['xmin']) / 2
    ePosy = (aCars[ei]['ymax'] + aCars[ei]['ymin']) / 2
    
    dis = math.sqrt((sPosx-ePosx)**2+(sPosy-ePosy)**2)
    print("dis=",dis/(ei-si)) 
    return dis/(ei-si) 

def checkOL(car, obj):
    if(car['xmin'] > obj['xmax'] or obj['xmin'] > car['xmax']):
        print('return x',car['xmin'],obj['xmax'],obj['xmin'],car['xmax'])
        return False
    if(car['ymin'] > obj['ymax'] or obj['ymin'] > car['ymax']):
        print('return y',car['ymin'],obj['ymax'],obj['ymin'],car['ymax'])
        return False    
    return True
    
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
            
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1['xmin'], box1['xmax']], [box2['xmin'], box2['xmax']])
    intersect_h = _interval_overlap([box1['ymin'], box1['ymax']], [box2['ymin'], box2['ymax']])

    intersect = intersect_w * intersect_h

    w1, h1 = box1['xmax']-box1['xmin'], box1['ymax']-box1['ymin']
    w2, h2 = box2['xmax']-box2['xmin'], box2['ymax']-box2['ymin']

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union
    
def infinite_infer_run():
    global lastForkliftAssetId
    
    """ Entry point of the lambda function"""
    try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'
        output_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
                      7 : 'car', 8 : 'cat', 9 : 'chair', 10 : 'cow', 11 : 'dinning table',
                      12 : 'dog', 13 : 'horse', 14 : 'cars', 15 : 'person',
                      16 : 'pottedplant', 17 : 'sheep', 18 : 'sofa', 19 : 'train',
                      20 : 'tvmonitor'}
        output_map = {0: 'forklift', 1: 'pallet'}
        
        # The height and width of the training set images
        input_height = 300
        input_width = 300
        
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
        ret, model_path = mo.optimize('deploy_ssd_resnet50_512',
                              input_width, input_height)
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading object detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Object detection model loaded')
        # Set the threshold for detection
        detection_threshold = 0.25

        
        # Initialize parameters for speed test
        speed = 0
        iCars = 1      # index for batch 
        nCars = 5      # size of a batch
        aCars = []     # array of a batch
        fCars = False  # flag to see if a car in a frame
        fBatch = False # flag to see if a batch is building
        # Do inference until the lambda is killed.
        barcodeData = 'NA'
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            #print("ret=",ret, frame.shape)
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            print("results=",parsed_inference_results)
            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0]) / float(input_height)
            xscale = float(frame.shape[1]) / float(input_width)
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}
            pep_output = {}
            # reset max prob for cars to 0.0
            pCars = 0.0
            # reset car obstacle flag to false
            fCars = False
            fobst = False
            # Get the detected objects and probabilities
            for obj in parsed_inference_results[model_type]:
                if obj['prob'] > detection_threshold:
                    print("Object detected:",obj) 
                    # Add bounding boxes to full resolution frame
                    xmin = int(xscale * obj['xmin'])
                    ymin = int(yscale * obj['ymin'])
                    xmax = int(xscale * obj['xmax'])
                    ymax = int(yscale * obj['ymax'])
                    # find a car label and larger than previous car labels
                    # include label 7(car) and 14(mobobikes)
                    if obj['label'] == 0 and obj['prob'] > pCars:
                        msg = {} 
                        msg['xmin'] = xmin
                        msg['xmax'] = xmax
                        msg['ymin'] = ymin
                        msg['ymax'] = ymax
                        pCars = obj['prob']
                        fCars = True
                        
                    # Found obstacles   
                    if obj['label'] == 1:
                        obst = {}
                        obst['xmin'] = int(xscale * obj['xmin'])-2
                        obst['ymin'] = int(yscale * obj['ymin'])-2
                        obst['xmax'] = int(xscale * obj['xmax'])+2 
                        obst['ymax'] = int(yscale * obj['ymax'])+2
                        fobst = True
                        
                    # barcode detection
                    # loop over the detected barcodes
                    barcodes = pyzbar.decode(frame)
                    print('found {} barcodes'.format(len(barcodes)))
                    
                    for barcode in barcodes:
                    	# extract the bounding box location of the barcode and draw the
                    	# bounding box surrounding the barcode on the image
                    	(x, y, w, h) = barcode.rect
                    	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    	# the barcode data is a bytes object so if we want to draw it on
                    	# our output image we need to convert it to a string first
                    	barcodeData = barcode.data.decode("utf-8")
                    	barcodeType = barcode.type
                    	
                    	lastForkliftAssetId = barcodeData[:10]
                    	if 'forklift' in lastForkliftAssetId:
                    	    print('lastForkliftAssetId: ' + lastForkliftAssetId)
                    	else:
                    	    lastForkliftAssetId = 'forklift 1'
                    	    print('Default lastForkliftAssetId: ' + lastForkliftAssetId)
                    	# draw the barcode data and barcode type on the image
                    	text = "{} ({})".format(barcodeData, barcodeType)
                    	cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    		0.5, (0, 0, 255), 2)
                    	# print the barcode type and data to the terminal
                    	print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.rectangle method.
                    # Method signature: image, point1, point2, color, and tickness.
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 10)
                    # Amount to offset the label/probability text above the bounding box.
                    text_offset = 15
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.putText method.
                    # Method signature: image, text, origin, font face, font scale, color,
                    # and tickness
                    cv2.putText(frame, "{}: {:.2f}%".format(output_map[obj['label']],
                                                               obj['prob'] * 100),
                                (xmin, ymin-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
                    # Store label and probability to send to cloud
                    cloud_output[output_map[obj['label']]] = obj['prob']
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            
            if(fBatch):
                if(not fCars): 
                    aCars.append({})
                else:
                    aCars.append(msg)
                iCars += 1
                if(iCars == nCars):
                    speed = processCarsBatch(aCars, nCars)
                    client.publish(topic=iot_topic, payload='Speed for forklift is: {}'.format(speed))
                    fBatch = False
            else:
                if(fCars):
                    fBatch = True
                    iCars = 1
                    aCars = [msg]
                    
            if  fCars and fobst:   
                print('.....Found Forklift....')
                if bbox_iou(msg, obst) > 0.1 and speed > 100:
                    print('...COLLISION....')
                    pep_output['ts'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    pep_output['source_type'] = 'camera'
                    pep_output['asset_id'] = lastForkliftAssetId 
                    #pep_output['operator_id'] = '0'
                    if speed > 200:
                        pep_output['event_type'] = 'high impact'
                    else:
                        pep_output['event_type'] = 'low impact'
                    pep_output['event_info'] = 'Impact speed '+str(speed)
                    print('Publish pep_output',pep_output)
                    client.publish(topic=pep_topic.replace('XXXX', str(lastForkliftAssetId)), payload=json.dumps(pep_output))
            cloud_output['iCars'] = iCars
            cloud_output['aCars'] = aCars
            # Send results to the cloud
            if(fBatch):
                client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
                
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

infinite_infer_run()

