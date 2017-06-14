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



class Bebop():

    def __init__(self):
        rospy.init_node('parrot_plan', anonymous=True)

        # For GPS
        #self.quad_planstart = rospy.Publisher('/bebop/autoflight/start', std_msgs.msg.String,queue_size=1)
        #self.quad_planstop = rospy.Publisher('/bebop/autoflight/stop', std_msgs.msg.String,queue_size=1)


        # Manual flight plan
        self.quad_takeoff = rospy.Publisher('/bebop/takeoff', std_msgs.msg.Empty,queue_size=1)
        self.quad_land = rospy.Publisher('/bebop/land', std_msgs.msg.Empty,queue_size=1)
        self.quad_emergency = rospy.Publisher('/bebop/reset', std_msgs.msg.Empty,queue_size=1)
        self.quad_vel = rospy.Publisher('/bebop/cmd_vel', geometry_msgs.msg.Twist,queue_size=1)

        self.camera = Camera(topic='/bebop/image_raw')

        self.command = geometry_msgs.msg.Twist()

        COMMAND_PERIOD = 100 # milliseconds
        self.commandTimer = rospy.Timer(rospy.Duration(COMMAND_PERIOD / 1000.0), self.SendCommand)


    def takeoff(self):
        self.quad_takeoff.publish(std_msgs.msg.Empty())
        self.status = 1

    def land(self):
        self.quad_land.publish(std_msgs.msg.Empty())
        self.status = 0

    def emergency(self):
        self.quad_emergency.publish(std_msgs.msg.Empty())
        self.status = 0

    def set_command(self, roll=0, pitch=0, yaw_velocity=0, z_velocity=0):
        self.command.linear.x  = pitch
        self.command.linear.y  = roll
        self.command.linear.z  = z_velocity
        self.command.angular.z = yaw_velocity

    def hover(self):
        self.set_command()   

    def SendCommand(self,event):
        # The previously set command is then sent out periodically if the drone is flying
        if self.status == 1:
            self.quad_vel.publish(self.command)

def parrot_plan():
    
    bebop = Bebop()
    predictor = Predictor()

    rate = rospy.Rate(10) # 10hz
    interval_rate = rospy.Rate(1) # 1 Hz-> 1 second

    bebop.takeoff()

    # Takeoff and increase height
    bebop.set_command(roll=0, pitch=0, yaw_velocity=0, z_velocity=1)    
    height = 5
    for i in range(height):
        rate.sleep()

    bebop.hover()

    (start_time, path_time, count) = (rospy.Time.now().to_sec(), 0, 0)     

    while not rospy.is_shutdown(): 
        if path_time < 3: 

            # For GPS
            # filename = std_msgs.msg.String()
            # filename.data = 'broadcenter.mavlink' # Uploaded via filezilla to bebop drone's flightplans folder
            # quad_planstart.publish(filename)
            
            # Manual flight plan to fly forward.
            bebop.set_command(roll=0, pitch=1, yaw_velocity=0, z_velocity=0)
            
        elif path_time < 30:
            #stop GPS autoflight
            #quad_planstop.publish(std_msgs.msg.Empty())
            
            # Stop flying forward
            bebop.hover()

        elif path_time < 500: 

            # Fixate on corner
            
            img = bebop.camera.getFrame() # Retrieve image from camera
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

            dX = dX / np.array(img.shape[1], img.shape[0]) 

            kp = 0.1  #proportional feedback (meters/(second*pixel))
            y_vel = -kp * dX[0] # linear.y>0 to move left
            z_vel = -kp * dX[1] # linear.z>0 to ascend

            bebop.set_command(roll=y_vel, pitch=0, yaw_velocity=0, z_velocity=z_vel)

            interval_rate.sleep()
            # revert to hover after 1 sec adjustment
            bebop.hover()

        else: # land after 500sec
            bebop.land() 
            break

        path_time = rospy.Time.now().to_sec() - start_time


if __name__ == '__main__':
    try:
        parrot_plan()
    except rospy.ROSInterruptException:
        pass
