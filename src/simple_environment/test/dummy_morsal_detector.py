#!/usr/bin/env python
import json, numpy, rospy
from std_msgs.msg import String
from StringIO import StringIO

def publisher():

    pub = rospy.Publisher('/perception/morsel_detection', String, queue_size=1)
    rospy.init_node('morsal_detector')
    rate = rospy.Rate(10) # 10 hz
    while not rospy.is_shutdown():
        morsal_in_camera = numpy.array([0.1, 0., 0.25])
        morsal_noise = numpy.random.normal(0, 0.01, 3)

        morsal_in_camera += morsal_noise
        msg = {'pts3d': morsal_in_camera.tolist()}

        pub.publish(json.dumps(msg))
        rate.sleep()

if __name__ == '__main__':
    
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
