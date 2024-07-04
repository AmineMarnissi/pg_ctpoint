#!/usr/bin/env python

import math
import rospy
import numpy as np
from sort import sort
from sensor_msgs.msg import Image
from sort_track.msg import IntList
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from detection_msgs.msg import BoundingBoxes
from std_msgs.msg import String
import time
import logging

# Configure logging
logging.basicConfig(filename='tracker.log', level=logging.INFO, format='%(asctime)s %(message)s')

# List of detection classes to be tracked
list_detections = ["person", "truck", "train", "car", "bus", "motorcycle", "bicycle"]

# Global variables
old_trackers = []
epsilon = 10
tracker_dict = {}
count_pass = 0
count_not_pass = 0
red_box = np.array([270, 100, 100, 50])
increment = 0

def euclidean(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_parameters():
    """
    Gets the necessary parameters from the parameter server.
    Returns:
        Tuple containing parameters.
    """
    camera_topic = rospy.get_param("~camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    tracker_topic = rospy.get_param("~tracker_topic")
    cost_threshold = rospy.get_param("~cost_threshold")
    min_hits = rospy.get_param("~min_hits")
    max_age = rospy.get_param("~max_age")
    return camera_topic, detection_topic, tracker_topic, cost_threshold, max_age, min_hits

def iou(bbox1, bbox2):
    """Compute intersection over union between two bounding boxes."""
    bbox1_tl, bbox1_br = bbox1[:2], bbox1[:2] + bbox1[2:]
    bbox2_tl, bbox2_br = bbox2[:2], bbox2[:2] + bbox2[2:]

    tl = np.maximum(bbox1_tl, bbox2_tl)
    br = np.minimum(bbox1_br, bbox2_br)
    wh = np.maximum(0., br - tl)
    area_intersection = np.prod(wh)
    area_bbox1 = np.prod(bbox1[2:])
    area_bbox2 = np.prod(bbox2[2:])
    area_union = area_bbox1 + area_bbox2 - area_intersection

    return area_intersection / area_union

def callback_found(data):
    """Callback function to handle found detections."""
    global increment
    found_detections = data.count
    if found_detections == 0:
        increment += 1
    else:
        increment = 0

    if increment > 5:
        logging.info("No detections found for a while, passing critic point")
        pub_critic.publish("pass")
        increment = 0

def callback_det(data):
    """Callback function to handle detections and update trackers."""
    global detections, trackers, old_trackers, count_pass, count_not_pass

    detections = []
    trackers = []
    track = []
    print(data)


    # Collect detections from the message
    for box in data.bounding_boxes:
        if box.Class in list_detections:
            logging.info(f"Class detected: {box.Class}, with confidence: {round(box.probability, 2)}")
            detections.append([box.xmin, box.ymin, box.xmax, box.ymax, round(box.probability, 2)])
    detections = np.array(detections)

    # Update trackers
    start_track_time = time.time()
    trackers = tracker.update(detections)
    trackers = np.array(trackers, dtype='int')
    end_track_time = time.time()
    logging.info(f"Time of tracking: {end_track_time - start_track_time}")

    # Check movement and update tracker list
    if old_trackers.any() and trackers.any():
        for old_t in old_trackers:
            for t in trackers:
                if old_t[4] == t[4]:  # Compare tracker IDs
                    old_x_center = (old_t[0] + old_t[2]) / 2
                    old_y_center = (old_t[1] + old_t[3]) / 2
                    x_center = (t[0] + t[2]) / 2
                    y_center = (t[1] + t[3]) / 2
                    distance_euc = euclidean(old_x_center, old_y_center, x_center, y_center)
                    if euclidean(old_x_center, old_y_center, x_center, y_center) > 3.0:
                        logging.info(f"Distance of mvt en pix: {distance_euc}")
                        bb1 = np.array([old_t[0], old_t[1], old_t[2] - old_t[0], old_t[3] - old_t[1]])
                        bb2 = np.array([t[0], t[1], t[2] - t[0], t[3] - t[1]])
                        region_overlap= iou(bb1, bb2)
                        if region_overlap < epsilon:
                            logging.info(f"Overlap region for mvt: {region_overlap}")
                            track.append(t)

    if track:
        count_pass = 0
        count_not_pass += 1
    else:
        count_not_pass = 0
        count_pass += 1

    if count_pass >= 70:
        pub_critic.publish("pass")
        logging.info("No objects detected, passing")
        rospy.signal_shutdown("Pass threshold reached")
    else:
        pub_critic.publish("not pass")

    old_trackers = trackers
    
    def main():
        global tracker

        # Initialize ROS node
        rospy.init_node('ctpoint_track', anonymous=False)
        rate = rospy.Rate(10)

        # Get parameters
        camera_topic, detection_topic, tracker_topic, cost_threshold, max_age, min_hits = get_parameters()

        # Initialize SORT tracker
        tracker = sort.Sort(max_age=max_age, min_hits=min_hits)

        # Subscribe to detection topic
        rospy.Subscriber(detection_topic, BoundingBoxes, callback_det)

        # Create publisher for critic points
        global pub_critic
        pub_critic = rospy.Publisher('/Safety_robot_critic_point', String, queue_size=10)

        rospy.spin()

    if __name__ == '__main__':
        try:
            main()
        except rospy.ROSInterruptException:
            pass
