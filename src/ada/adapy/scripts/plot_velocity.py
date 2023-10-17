#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--time-padding', type=float, default=0.1)
	parser.add_argument('--time-steady', type=float, default=0.74)
	parser.add_argument('input_file', type=str)
	args = parser.parse_args()

	data = numpy.loadtxt(args.input_file)
	stamps, positions, velocities = data.T
	stamps -= stamps[0]

	velocity_limit = velocities[stamps > args.time_steady].mean()

	linear_mask = numpy.logical_and(
		stamps > args.time_padding,
	    stamps < args.time_steady - args.time_padding
	)
	b = velocities[linear_mask]
	A = numpy.column_stack((
		stamps[linear_mask],
		numpy.ones(len(b))
	))
	x, _, _, _ = numpy.linalg.lstsq(A, b)	

	acceleration_limit = x[0]

	print 'Velocity Limit:     {:.6f}'.format(abs(velocity_limit))
	print 'Acceleration Limit: {:.6f}'.format(abs(acceleration_limit))

	fig = plt.figure()
	axis_position = fig.add_subplot(2, 1, 1)
	axis_position.plot(stamps, positions, '-k')
	axis_position.set_ylabel('Position (rad)')

	axis_velocity = fig.add_subplot(2, 1, 2)
	axis_velocity.plot(stamps, velocities, '-b')
	axis_velocity.plot(stamps, [velocity_limit] * len(stamps), '-r')
	axis_velocity.plot(stamps, x[0] * stamps + x[1], '-g')
	axis_velocity.set_xlabel('Time (s)')
	axis_velocity.set_ylabel('Velocity (rad/s)')

	plt.show()
