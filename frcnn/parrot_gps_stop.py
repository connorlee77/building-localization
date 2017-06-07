#!/usr/bin/env python

import rospy
import math
import geometry_msgs.msg
import std_msgs.msg

    
def parrot_plan():
    rospy.init_node('parrot_stop', anonymous=True)
    quad_emergency = rospy.Publisher('/bebop/reset', std_msgs.msg.Empty,queue_size=1)
    quad_emergency.publish(std_msgs.msg.Empty())

if __name__ == '__main__':
    try:
        parrot_plan()
    except rospy.ROSInterruptException:
        pass
