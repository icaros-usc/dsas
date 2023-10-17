#!/usr/bin/env python

# Copyright (c) 2013, Carnegie Mellon University
# All rights reserved.
# Authors: Michael Koval <mkoval@cs.cmu.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of Carnegie Mellon University nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rospkg, os, sys, distutils.version

def is_catkin():
    distro = os.environ.get('ROS_DISTRO', '')
    if distro:
        # Clever hack courtesy of Chris Dellin.
        return ord(distro[0].lower()) >= ord('g')
    else:
        raise ValueError('Environment variable ROS_DISTRO is not set.')

if is_catkin():
    def export():
        pass # do nothing
else:
    def append_to_env(variable, new_path, separator=':'):
        paths = os.environ.get(variable, '')
        if paths:
            paths_list = paths.split(separator)
        else:
            paths_list = list()

        # Check if we're adding a duplicate entry.
        duplicate = False
        new_canonical_path = os.path.realpath(new_path)
        for path in paths_list: 
            canonical_path = os.path.realpath(path)
            if canonical_path == new_canonical_path:
                duplicate = True
                break

        # Add the new path to the environmental variable.
        if not duplicate:
            paths_list.insert(0, new_path)
            paths = separator.join(paths_list)
            os.environ[variable] = paths

        return paths 

    def export_paths(pkg, package_name):

        packages = list()

        # first get all the required packages
        try:
            manifest = pkg.get_manifest(package_name)
            required_packages = pkg.get_depends(package_name)
            packages.extend(required_packages)
        except rospkg.ResourceNotFound:
            return False, [package_name]

        # next get all the optional packages
        optional_packages = manifest.get_export('openrave', 'optional')
        packages.extend(optional_packages)

        # Process the OpenRAVE export tags. 
        for plugin_path in manifest.get_export('openrave', 'plugins'):
            plugin_paths = append_to_env('OPENRAVE_PLUGINS', plugin_path)
            
        for data_path in manifest.get_export('openrave', 'data'):
            data_paths = append_to_env('OPENRAVE_DATA', data_path)

        for database_path in manifest.get_export('openrave', 'database'):
            database_paths = append_to_env('OPENRAVE_DATABASE', database_path)
            
        for python_path in manifest.get_export('python', 'path'):
            sys.path.append(python_path)

        # Add the appropriate directories to PYTHONPATH.
        # TODO: This is a hack. Should I parsing a Python export instead?
        sys.path.append(pkg.get_path(package_name) + '/src')

        # now dive into the subpackages
        packages = set(packages)
        missing_packages = list()
        for package in packages:
            success, missing = export_paths(pkg, package)
            missing_packages.extend(missing)
            if not success:
                # check if this package is required
                if package in required_packages:
                    return False, missing_packages

        return True, missing_packages

    def export(package_name):
        pkg = rospkg.RosPack()
        manifest = pkg.get_manifest(package_name)

        # Required dependencies.
        success, missing_packages = export_paths(pkg, package_name)
        if not success:
            raise Exception('Unable to load required dependencies.')

        if missing_packages:
            missing_packages.sort()
            print 'Missing optional dependencies: %s' % ' '.join(missing_packages)
        return missing_packages
