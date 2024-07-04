#!/usr/bin/env python

"""
ROS node to track objects using SORT TRACKER and YOLOv3 detector (darknet_ros)
Takes detected bounding boxes from darknet_ros and uses them to calculate tracked bounding boxes
Tracked objects and their ID are published to the sort_track node
No delay here
"""
import math
import rospy
import numpy as np
from sort import sort 
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from sort_track.msg import IntList
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from detection_msgs.msg import BoundingBoxes
from std_msgs.msg import String
import logging
from bs_tracking_ros.srv import CountPass, CountPassRequest, CountPassResponse


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure logger is set to debug level

file_handler = logging.FileHandler('/home/amine/pg_commande/catkin_ws/src/pg_ctpoint/src/tracker.log')
file_handler.setLevel(logging.DEBUG)  # Ensure handler is set to debug level

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Test logging
logger.debug("Logger has been configured and this is a test message.")

# Set up console logging for debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Class names
names = ["person", "truck", "train", "car", "bus", "motorcycle", "bicycle"]

count_pass = 0
count_not_pass = 0
not_safe_site = 0

def get_parameters():
    """
    Gets the necessary parameters from .yaml file
    Returns dictionary of parameters
    """
    params = {
        "camera_topic": rospy.get_param("~camera_topic", "/camera/image_raw"),
        "detection_topic": rospy.get_param("~detection_topic", "/darknet_ros/bounding_boxes"),
        "tracker_topic": rospy.get_param('~tracker_topic', "/tracked_objects"),
        "cost_threshold": rospy.get_param('~cost_threshold', 0.3),
        "min_hits": rospy.get_param('~min_hits', 3),
        "max_age": rospy.get_param('~max_age', 1)
    }
    return params

def count_pass_client():
    rospy.wait_for_service('count_pass_service')
    
    try:
        count_pass_service = rospy.ServiceProxy('count_pass_service', CountPass)
        request = CountPassRequest()
        response = count_pass_service(request)
        rospy.loginfo("Count Pass Value: %d", response.count_pass)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)
    return response.count_pass    

def callback_det(data):
    global detections
    global trackers
    global count_pass
    global count_not_pass
    global not_safe_site
    
    
    detections = np.empty((0, 6))
    for box in data.bounding_boxes:
        detections = np.vstack((detections, [box.xmin, box.ymin, box.xmax, box.ymax, round(box.probability, 4), names.index(box.Class)]))
        logger.info(f"Class detected: {box.Class}, with confidence: {round(box.probability, 2)}")
    
    trackers = tracker.update(detections)

    v_magnitudes = trackers[:, 5] 
    angles = trackers[:, 6]
    
    indices_greater_than_1 = np.where(v_magnitudes > 1.0)
    magnitudes_greater_than_1 = v_magnitudes[indices_greater_than_1]

    if len(magnitudes_greater_than_1) > 0:
        count_pass = 0 
        count_not_pass += 1
        rospy.loginfo("Count not pass: %d", count_not_pass)
    else:
        count_not_pass = 0
        count_pass += 1
        rospy.loginfo("Count pass: %d", count_pass)
    
    if count_pass >= 70:
        pub_critic.publish("pass")
        logger.info("Published 'pass' message.")
        logger.error("Exceeded maximum pass count. Stopping operation.")
        rospy.signal_shutdown("Exceeded maximum pass count")
    else:
        pub_critic.publish("not pass")
        not_safe_site += 1
        logger.info("Published 'not_safe_site':  %d sec", not_safe_site)
    
    if not_safe_site>=10:
        time_BS_not_detect = count_pass_client()
        logger.info("Published 'time BS not detect':  %d sec", time_BS_not_detect)
        if time_BS_not_detect>10:
            pub_critic.publish("pass")
            logger.info("Published 'pass' message by BS not detect service.")
            logger.error("Exceeded maximum pass count. Stopping operation.")
            rospy.signal_shutdown("Exceeded maximum pass count")
        else:
            logger.info("Not pass BS detect object.")
            not_safe_site = 0

def main():
    global tracker
    global msg
    global pub_critic
    
    rospy.init_node('ctpoint_track', anonymous=False)
    logger.info("ROS node initialized.")
    
    msg = IntList()
    params = get_parameters()
    
    tracker = sort.Sort(max_age=params['max_age'], min_hits=params['min_hits']) # Create instance of the SORT tracker
    
    pub_critic = rospy.Publisher('/Safety_robot_critic_point', String, queue_size=10)
    
    sub_detection = rospy.Subscriber(params['detection_topic'], BoundingBoxes, callback_det)
    
    rospy.spin()
    logger.info("ROS spinning...")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        logger.error("ROSInterruptException caught.")
        pass
