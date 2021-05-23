import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
#FLAGS to use in terminal
flags.DEFINE_string('video', './data/video/test.mp4','path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('def_line','center','Crossing line alignment')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5   #To check if tracker belongs to the same object in previous frame
    nn_budget = None            #Used to save feature vectors (by default 100)
    nms_max_overlap = 0.8      #Threshold for nms

    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'                              #Pretrained CNN
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)             #Encoder for Feature Generation
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)    #Association matrix using defined cosine distance & nn_budget
    tracker = Tracker(metric)       #Initialize tracker

    #Get class names from custom .names file
    class_names = [c.strip() for c in open('./data/labels/obj.names')]
    #Show in terminal
    logging.info('classes loaded')
    #Initialize YoloV3 with number of classes
    yolo = YoloV3(classes=len(class_names))
    #Load trained custom weights
    yolo.load_weights('./weights/yolov3-custom.tf')
    #Show in terminal
    logging.info('weights loaded')

    #FLAGS.video = 0 for webcam
    #vid = cv2.VideoCapture(int(FLAGS.video))
    vid = cv2.VideoCapture(FLAGS.video)
    #out = None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #'XVID' for .avi video
    codec = cv2.VideoWriter_fourcc(*'XVID')

    #Create output video configurations based on input config
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    list_file = open('detection.txt', 'w')
    frame_index = -1

    fps = 0.0                       #To display fps

    total_mask_counter=[]           #To keep count of total people wearing masks
    total_no_mask_counter=[]        #To keep count of total people not wearing masks
    total_incorrect_counter=[]      #To keep count of total people wearing masks incorrectly

    mask_cross_counter=[]           #To keep count of number of mask wearing people crossing line
    no_mask_cross_counter=[]        #To keep count of number of people not wearing masks crossing line
    incorrect_cross_counter=[]      #To keep count of number of people wearing masks incorrectly crossing line

    while True:
        _, img = vid.read()

        #When video has ended break out of loop
        if img is None:
            logging.warning("Empty Frame/Video Complete")
            break
        #Read frame image and convert color from BGR to RGB
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Add batch size as another dimension to image matrix
        img_in = tf.expand_dims(img_in, 0)
        #Resize image to YoloV3 default size
        img_in = transform_images(img_in, 416)

        t1 = time.time()
        #boxes-> shape(1,100,4) : x,y,w,h
        #scores-> shape(1,100)  :Confidence score
        #nums-> shape(1,)       :total objects detected
        boxes, scores, classes, nums = yolo.predict(img_in) #Pass image into Yolo instance
        classes = classes[0]                                #Take top class detected
        names = []                                          #To keep class name record
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        #ROI projection to initial image
        converted_boxes = convert_boxes(img, boxes[0])      #Make use of first row of boxes
        #Generates feature vectors for detected objects
        features = encoder(img, converted_boxes)
        #Compilations of all the vectors of detection matrix
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        # run non-maximum suppresion
        boxs = np.array([d.tlwh for d in detections])                   #TopLeftxy,Width,Height of box
        scores = np.array([d.confidence for d in detections])           #Condidence scores
        classes = np.array([d.class_name for d in detections])          #Class names
        #Tells us which boxes should be kept/discarded
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]   #Removes redundencies

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        height, width, _ = img.shape
        if FLAGS.def_line == 'center':
            width=width/2
        if FLAGS.def_line == 'right':
            width=width-width/3
        if FLAGS.def_line == 'left':
            width=width/3
        cv2.line(img,(int(width),0),(int(width),height),(255,0,0),thickness=2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:     #if tracker doesnt exist or is not updated continue to next tracker
                continue
            bbox = track.to_tlbr()          #Get bbox coord (TopLeft (x,y),BottomRight (x,y))
            class_name = track.get_class()  #Get corresponding classes

            if class_name=='Mask':
                color=(0,255,0)
                total_mask_counter.append(int(track.track_id))
            elif class_name=='No_Mask':
                color=(0,0,255)
                total_no_mask_counter.append(int(track.track_id))
            else:
                color=(0,0,255)
                total_incorrect_counter.append(int(track.track_id))
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, color,2)
            #####################################
            res=65
            center_x = int(((bbox[0])+(bbox[2]))/2)
            if center_x<=(int(width+res)) and center_x>=(int(width-res)):
                if class_name=='Mask':
                    mask_cross_counter.append(int(track.track_id))
                elif class_name=='No_Mask':
                    no_mask_cross_counter.append(int(track.track_id))
                else:
                    incorrect_mask_counter.append(int(track.track_id))
        font_scale=0.75
        cv2.putText(img, "Mask Person Crossed: " + str(len(set(mask_cross_counter))), (800,height-90), 0,font_scale, (0,255,255), 2)
        cv2.putText(img, "NoMask Person Crossed: " + str(len(set(no_mask_cross_counter))), (800,height-50), 0, font_scale, (0,255,255), 2)
        cv2.putText(img, "Incorrect Mask Person Crossed: " + str(len(set(incorrect_cross_counter))), (800,height-10), 0, font_scale, (0,255,255), 2)

        cv2.putText(img, "Total Mask Person Count: " + str(len(set(total_mask_counter))), (0,height-90), 0, font_scale, (0,255,255), 2)
        cv2.putText(img, "Total Non Mask Person Count: " + str(len(set(total_no_mask_counter))), (0,height-50), 0, font_scale, (0,255,255), 2)
        cv2.putText(img, "Total Incorrect Person Count: " + str(len(set(total_incorrect_counter))), (0,height-10), 0, font_scale, (0,255,255), 2)
            #####################################
        #Display FPS
        fps  = (fps + (1./(time.time()-t1))) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (550, height-50),3, font_scale, (0,0,255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
