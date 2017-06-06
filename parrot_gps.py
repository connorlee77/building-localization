#!/usr/bin/env python

import rospy
import math
import geometry_msgs.msg
import std_msgs.msg

def flightplan(): # Not used, but contains intended waypoint file (QGC's XML format)
	plan = std_msgs.msg.String()
	plan.data = 'QGC WPL 120\n0\t0\t10\t16\t5.000000\t0.100000\t0.000000\t0.000000\t34.139171\t-118.127511\t2.000000\t1'
	# format QGroundControl WayPointList? Version Index CoordFrame Command HoldTime Radius(m) OrbitRadius CamAngle Lat Long Altitude AutoContinue
	# see <qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format>
	# see <https://pixhawk.ethz.ch/mavlink/#MISSION_ITEM

	
def parrot_plan():
	rospy.init_node('parrot_plan', anonymous=True)
	quad_planstart = rospy.Publisher('/bebop/autoflight/start', std_msgs.msg.String,queue_size=1)
	quad_planstop = rospy.Publisher('/bebop/autoflight/stop', std_msgs.msg.String,queue_size=1)
	quad_takeoff = rospy.Publisher('/bebop/takeoff', std_msgs.msg.Empty,queue_size=1)
	quad_land = rospy.Publisher('/bebop/land', std_msgs.msg.Empty,queue_size=1)

	rate = rospy.Rate(10) # 10hz
	count = 0
	while count < 70: # Takeoff
		if count < 20:
			quad_takeoff.publish(std_msgs.msg.Empty())
		elif count < 50:
			cmd = geometry_msgs.msg.Twist()
			cmd.linear.z = 1.0
			quad_vel.publish(cmd)
		else:
			cmd = geometry_msgs.msg.Twist()
			quad_vel.publish(cmd)
		count += 1
		rate.sleep()

	(start_time, path_time, count) = (rospy.Time.now().to_sec(), 0, 0)     

	while not rospy.is_shutdown(): 
		if path_time < 20: # 20sec of flying towards GPS coords (see flightplan()), currently set in the middle of Broad Field
			filename = std_msgs.msg.String()
			filename.data = 'broadcenter.mavlink' # Uploaded via filezilla to bebop drone's flightplans folder
				quad_planstart.publish(filename)
			# find the filename and put it in String.data, or change filename to flightplan.mavlink (default)
			# use filezilla to find/copy mavlink (waypoint) files
			# could also use autoflight/start -> /pause (Empty) -> /start (resumes from most recent waypoint)
			# then, the waypoint file can dictate both takeoff and landing (no need for time counting)
		


		elif path_time < 500: # Fixate on corner
			######
			######
			#      TODO (image processing)
			######
			######
		




		else: # LAND
			while count < 20: #land after 20sec
				if count == 1:
					quad_planstop.publish(std_msgs.msg.Empty())
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
