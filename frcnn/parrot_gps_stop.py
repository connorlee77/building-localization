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
    rospy.init_node('parrot_stop', anonymous=True)
    quad_land = rospy.Publisher('/bebop/land', std_msgs.msg.Empty,queue_size=1)
    quad_vel = rospy.Publisher('/bebop/cmd_vel', std_msgs.msg.Empty,queue_size=1)

    camera = Camera(topic='/bebop/image_raw')
    predictor = Predictor()

    rate = rospy.Rate(10) # 10hz     

    while not rospy.is_shutdown(): 
        cmd = std_msgs.msg.Empty()
        quad_land.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        parrot_plan()
    except rospy.ROSInterruptException:
        pass
