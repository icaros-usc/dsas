#!/usr/bin/env python
import adapy
import argparse
import numpy
import rospy
from adapy.controller_client import ControllerManagerClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

running = False
stamps = []
positions = []
velocities = []

def jointstate_callback(jointstate_msg):
	if running:
		i = jointstate_msg.name.index("mico_joint_"+joint_name[-1])
		stamps.append(jointstate_msg.header.stamp.to_sec())
		positions.append(jointstate_msg.position[i])
		velocities.append(jointstate_msg.velocity[i])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('joint_name', type=str)
	args = parser.parse_args()

	rospy.init_node('test_velocity', anonymous=True)

	velocity = 10.0
	duration = 1.0
	joint_name = args.joint_name
	output_path = '{:s}.csv'.format(joint_name)

	controller_name = 'vel_{:s}_controller'.format(joint_name)
	command_topic_name = '{:s}/command'.format(controller_name)
	jointstate_topic_name = 'joint_states'

	controller_client = ControllerManagerClient('/controller_manager')
	velocity_controller = controller_client.request(controller_name)
	velocity_publisher = rospy.Publisher(command_topic_name, Float64)
	jointstate_subscriber = rospy.Subscriber(jointstate_topic_name, JointState, jointstate_callback)

	with velocity_controller:
		rospy.sleep(0.5) # mystery race condition in ros_control

		rospy.loginfo('Moving %s at %.2f rad/s for %.2f s.',
			joint_name, velocity, duration
		)
		velocity_publisher.publish(velocity)
		running = True

		rospy.sleep(duration)

		rospy.loginfo('Stopping!')
		running = False
		velocity_publisher.publish(0.)

	data = numpy.column_stack((stamps, positions, velocities))
	numpy.savetxt(output_path, data)
	rospy.loginfo('Saved %d data points to "%s".', data.shape[0], output_path)
