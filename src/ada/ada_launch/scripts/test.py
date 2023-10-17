#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import atexit
import os
import time
import openravepy
import roslib

# sweet path magic
path_append = lambda e,k,p: e.update({k: ((e[k]+':') if k in e else '') + p}) 
ormaj, ormin, _ = map(int,openravepy.__version__.split('.'))
for libpath in os.environ['LD_LIBRARY_PATH'].split(':')+['/usr/lib']:
   path_append(os.environ, 'OPENRAVE_PLUGINS',
      '{}/openrave-{}.{}'.format(libpath, ormaj, ormin))
path_append(os.environ, 'OPENRAVE_DATA',
   '{}/ordata'.format(roslib.packages.get_pkg_dir('ada_description')))

# load an openrave environment
openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
atexit.register(openravepy.RaveDestroy)
e = openravepy.Environment()
atexit.register(e.Destroy)

e.SetViewer('qtcoin')

r = e.ReadRobotXMLFile('robots/mico-modified.robot.xml')
e.Add(r)

c = openravepy.RaveCreateController(e,'roscontroller openrave / 1')
r.SetController(c,range(8),0)

#r.SetActiveDOFs(range(6))

time.sleep(1)

# make a sweet trajectory
raw_input('Press [Enter] to send trajectory ...')
t = openravepy.RaveCreateTrajectory(e, '')
t.Init(r.GetActiveConfigurationSpecification())
print(t.GetConfigurationSpecification())
t.Insert(0, r.GetActiveDOFValues())
if r.GetActiveDOFValues()[5] < 0.5:
   t.Insert(1, [0.0, -1.57, 1.57, 0.0, 0.0, 1.0, 0.0, 0.0])
else:
   t.Insert(1, [0.0, -1.57, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0])
openravepy.planningutils.RetimeActiveDOFTrajectory(t, r)
r.GetController().SetPath(t)

raw_input('Press [Enter] to quit ...')


