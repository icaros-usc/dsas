#!/usr/bin/env python
"""
Provides a simple console that sets up basic functionality for 
using AdaPy and openravepy.
"""

import IPython
import adapy
import argparse
import logging
import numpy
import openravepy
import rospy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='utility script for loading AdaPy')
    parser.add_argument('-s', '--sim', action='store_true',
                        help='simulation mode')
    parser.add_argument('-v', '--viewer', nargs='?', const=True,
                        help='attach a viewer of the specified type')
    parser.add_argument('--env-xml', type=str,
                        help='environment XML file; defaults to an empty environment')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug logging')
    args = parser.parse_args()

    openravepy.RaveInitialize(True)
    openravepy.misc.InitOpenRAVELogging()

    if args.debug:
        openravepy.RaveSetDebugLevel(openravepy.DebugLevel.Debug)

    #if not args.sim:
    rospy.init_node('adapy', anonymous=True)

    env, robot = adapy.initialize(
        sim=args.sim,
        attach_viewer=args.viewer,
        env_path=args.env_xml
    )

    IPython.embed()
