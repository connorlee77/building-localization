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
    count = 0
    while count < 70: # Takeoff
        if count < 20:
            quad_takeoff.publish(std_msgs.msg.Empty())
            print "takeoff"
        elif count < 50:
            cmd = geometry_msgs.msg.Twist()
            cmd.linear.z = 0.25
            quad_vel.publish(cmd)
            print "move up"
        else:
            print "hover"
            cmd = geometry_msgs.msg.Twist()
            quad_vel.publish(cmd)
        count += 1
        rate.sleep()

    (start_time, path_time, count) = (rospy.Time.now().to_sec(), 0, 0)     

    while not rospy.is_shutdown(): 
        if path_time < 5: # 20sec of GPS flying
            print "flying to building"
            #see flightplan() for example GPS format, currently set in the middle of Broad Field
            # filename = std_msgs.msg.String()
            # filename.data = 'broadcenter.mavlink' # Uploaded via filezilla to bebop drone's flightplans folder
            # quad_planstart.publish(filename)

            cmd = geometry_msgs.msg.Twist()
            cmd.linear.x = 0.25
            quad_vel.publish(cmd)

            # find the filename and put it in String.data, or change filename to flightplan.mavlink (default)
            # use filezilla to find/copy mavlink (waypoint) files
            # could also use autoflight/start -> /pause (Empty) -> /start (resumes from most recent waypoint)
            # then, the waypoint file can dictate both takeoff and landing (no need for time counting)
        
        elif path_time < 30: #stop GPS autoflight
            #quad_planstop.publish(std_msgs.msg.Empty())
            cmd = geometry_msgs.msg.Twist()
            quad_vel.publish(cmd)
            print "done flying to building"

        elif path_time < 500: # Fixate on corner
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

                dist_to_target = np.linalg.norm(target - center)
                if dist_to_target < min_dist:
                    min_dist = dist_to_target
                    dX = center - corner_center

            print dX 
            cmd = geometry_msgs.msg.Twist()
            kp = 0.01  #proportional feedback (meters/(second*pixel))
            cmd.linear.y = -kp * dX[0] # linear.y>0 to move left
            cmd.linear.z = -kp * dX[1] # linear.z>0 to ascend
            quad_vel.publish(cmd)
            interval_rate = rospy.Rate(1) # 1 Hz-> 1 second
            interval_rate.sleep()
            cmd = geometry_msgs.msg.Twist() # revert to hover after 1 sec adjustment
            quad_vel.publish(cmd)
            

        else: #land after 500sec
            print "landing"
            while count < 20:
                if count == 1:
                    cmd = geometry_msgs.msg.Twist()
                    quad_vel.publish(cmd)
                quad_land.publish(std_msgs.msg.Empty())
                count += 1
                rate.sleep()

        path_time = rospy.Time.now().to_sec() - start_time
        rate.sleep()

if __name__ == '__main__':
    try:
        parrot_plan()
    except rospy.ROSInterruptException:
        pass
