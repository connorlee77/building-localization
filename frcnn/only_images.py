#!/usr/bin/env python

import rospy
import math
import geometry_msgs.msg
import std_msgs.msg

import cv2
import numpy as np 

from predictor import Predictor
from camera import Camera

def flightplan(): # Not used, but contains intended waypoint file (QGC's XML format)
    plan = std_msgs.msg.String()
    plan.data = 'QGC WPL 120\n0\t0\t10\t16\t5.000000\t0.100000\t0.000000\t0.000000\t34.139171\t-118.127511\t2.000000\t1'
    # format QGroundControl WayPointList? Version Index CoordFrame Command HoldTime Radius(m) OrbitRadius CamAngle Lat Long Altitude AutoContinue
    # see <qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format>
    # see <https://pixhawk.ethz.ch/mavlink/#MISSION_ITEM

    
def parrot_plan():
    rospy.init_node('parrot_plan', anonymous=True)
    #quad_planstart = rospy.Publisher('/bebop/autoflight/start', std_msgs.msg.String,queue_size=1)
    #quad_planstop = rospy.Publisher('/bebop/autoflight/stop', std_msgs.msg.String,queue_size=1)
    quad_takeoff = rospy.Publisher('/bebop/takeoff', std_msgs.msg.Empty,queue_size=1)
    quad_land = rospy.Publisher('/bebop/land', std_msgs.msg.Empty,queue_size=1)
    quad_vel = rospy.Publisher('/bebop/cmd_vel', geometry_msgs.msg.Twist,queue_size=1)

    camera = Camera(topic='/bebop/image_raw')
    predictor = Predictor()

    rate = rospy.Rate(10) # 10hz


    (start_time, path_time, count) = (rospy.Time.now().to_sec(), 0, 0)     

    while not rospy.is_shutdown(): 
        
        print "get corner"
        img = camera.getFrame() # Retrieve image from camera
        name = 'frame' + str(int(path_time)) + '.jpg' 

        corners = predictor.predict(img=img, img_name=name) # Predict corners

        center = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0]) # (x, y)            
        # Currently ignore probability. Only consider distance to center of camera
        min_dist = 100000000000
        dX = np.zeros(2) # Assume dX[0] is left/right translations, dX[1] is up/down translations; no yawing 
        for corner in corners:
            p, x1, y1, x2, y2 = corner
            corner_center = np.array([x2 - x1, y2 - y1])

            dist_to_target = np.linalg.norm(corner_center - center)
            if dist_to_target < min_dist:
                min_dist = dist_to_target
                dX = center - corner_center

        print dX 
        cmd = geometry_msgs.msg.Twist()
        kp = 0.01  #proportional feedback (meters/(second*pixel))
        cmd.linear.y = -kp * dX[0] # linear.y>0 to move left
        cmd.linear.z = -kp * dX[1] # linear.z>0 to ascend

        print cmd

        path_time = rospy.Time.now().to_sec() - start_time
        rate.sleep()

if __name__ == '__main__':
    try:
        parrot_plan()
    except rospy.ROSInterruptException:
        pass
