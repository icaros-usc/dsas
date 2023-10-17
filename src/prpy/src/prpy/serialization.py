import numpy
import openravepy
import logging
from .exceptions import UnsupportedTypeSerializationException

TYPE_KEY = '__type__'

serialization_logger = logging.getLogger('prpy.serialization')
deserialization_logger = logging.getLogger('prpy.deserialization')

# Serialization.
def serialize(obj):
    from numpy import ndarray
    from openravepy import Environment, KinBody, Robot, Trajectory
    from prpy.tsr import TSR, TSRChain

    NoneType = type(None)

    if isinstance(obj, numpy.floating):
        return float(obj)
    elif isinstance(obj, (numpy.signedinteger, numpy.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, (int, float, basestring, NoneType)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [ serialize(x) for x in obj ]
    elif isinstance(obj, dict):
        obj = { serialize(k): serialize(v) for k, v in obj.iteritems() }
        obj[TYPE_KEY] = dict.__name__
        return obj
    elif isinstance(obj, ndarray):
        return {
            TYPE_KEY: ndarray.__name__,
            'data': serialize(obj.tolist())
        }
    elif isinstance(obj, Environment):
        return {
            TYPE_KEY: Environment.__name__,
            'data': serialize_environment(obj)
        }
    elif isinstance(obj, KinBody):
        return {
            TYPE_KEY: KinBody.__name__,
            'name': obj.GetName()
        }
    elif isinstance(obj, Robot):
        return {
            TYPE_KEY: Robot.__name__,
            'name': obj.GetName()
        }
    elif isinstance(obj, KinBody.Link):
        return {
            TYPE_KEY: KinBody.Link.__name__,
            'name': obj.GetName(),
            'parent_name': obj.GetParent().GetName()
        }
    elif isinstance(obj, KinBody.Joint):
        return {
            TYPE_KEY: KinBody.Joint.__name__,
            'name': obj.GetName(),
            'parent_name': obj.GetParent().GetName()
        }
    elif isinstance(obj, Robot.Manipulator):
        return {
            TYPE_KEY: KinBody.Manipulator.__name__,
            'name': obj.GetName(),
            'parent_name': obj.GetParent().GetName()
        }
    elif isinstance(obj, Trajectory):
        return {
            TYPE_KEY: Trajectory.__name__,
            'data': obj.serialize(0)
        }
    elif isinstance(obj, TSR):
        return {
            TYPE_KEY: TSR.__name__,
            'data': obj.to_dict()
        }
    elif isinstance(obj, TSRChain):
        return {
            TYPE_KEY: TSRChain.__name__,
            'data': obj.to_dict()
        }
    else:
        raise UnsupportedTypeSerializationException(obj)

def serialize_environment(env, uri_only=False):
    return {
        'bodies': [ serialize_kinbody(body, uri_only=uri_only) for body in env.GetBodies() ],
    }

def serialize_environment_file(env, path, writer=None):
    if writer is None:
        import json
        writer = json.dump

    data = serialize_environment(env)

    if path is not None:
        with open(path, 'wb') as output_file:
            writer(data, output_file)
            serialization_logger.debug('Wrote environment to "%s".', path)

    return data

def serialize_kinbody(body, uri_only=False):

    uri = body.GetXMLFilename()
    
    if uri_only and not uri:
        serialization_logger.warn(
            'uri_only passed, but KinBody "{}"\'s GetXMLFilename()'
            ' returns an empty URI.'.format(
            body.GetName()))
        uri_only = False
    
    data = {
        'is_robot': body.IsRobot(),
        'name': body.GetName(),
        'uri': uri,
    }

    # Only add uri_only to the serialization if uri_only is set
    # so older deserializers work.
    if uri_only:
        data['uri_only'] = True
    
    if not uri_only:
        all_joints = []
        all_joints.extend(body.GetJoints())
        all_joints.extend(body.GetPassiveJoints())
        data['links'] = map(serialize_link, body.GetLinks())
        data['joints'] = map(serialize_joint, all_joints)
    
    data['kinbody_state'] = serialize_kinbody_state(body)

    if body.IsRobot():
        data.update(serialize_robot(body, uri_only=uri_only))

    return data

def serialize_robot(robot, uri_only=False):
    data = {
        'robot_state': serialize_robot_state(robot),
    }
    if not uri_only:
        data.update({
            'manipulators': map(serialize_manipulator, robot.GetManipulators()),
        })
    return data

def serialize_kinbody_state(body):
    data = {
        name: get_fn(body)
        for name, (get_fn, _) in KINBODY_STATE_MAP.iteritems()
    }

    link_transforms, dof_branches = body.GetLinkTransformations(True)
    data.update({
        'link_transforms': map(serialize_transform, link_transforms),
        'dof_branches': dof_branches.tolist(),
        'dof_values': body.GetDOFValues().tolist(),
    })

    return data

def serialize_robot_state(body):
    data = {
        name: get_fn(body)
        for name, (get_fn, _) in ROBOT_STATE_MAP.iteritems()
    }
    data['grabbed_bodies'] = map(serialize_grabbed_info, body.GetGrabbedInfo())
    return data

def serialize_link(link):
    data = { 'info': serialize_link_info(link.GetInfo()) }

    # Bodies loaded from ".kinbody.xml" do not have GeometryInfo's listed in
    # their LinkInfo class. We manually read them from GetGeometries().
    # TODO: This may not correctly preserve non-active geometry groups.
    data['info']['_vgeometryinfos'] = [
        serialize_geometry_info(geometry.GetInfo()) \
        for geometry in link.GetGeometries()
    ]
    return data

def serialize_joint(joint):
    return { 'info': serialize_joint_info(joint.GetInfo()) }

def serialize_manipulator(manipulator):
    return { 'info': serialize_manipulator_info(manipulator.GetInfo()) }

def serialize_with_map(obj, attribute_map):
    return {
        key: serialize_fn(getattr(obj, key))
        for key, (serialize_fn, _) in attribute_map.iteritems()
    }

def serialize_link_info(link_info):
    return serialize_with_map(link_info, LINK_INFO_MAP)
    
def serialize_joint_info(joint_info):
    return serialize_with_map(joint_info, JOINT_INFO_MAP)

def serialize_manipulator_info(manip_info):
    return serialize_with_map(manip_info, MANIPULATOR_INFO_MAP)

def serialize_geometry_info(geom_info):
    return serialize_with_map(geom_info, GEOMETRY_INFO_MAP)

def serialize_grabbed_info(grabbed_info):
    return serialize_with_map(grabbed_info, GRABBED_INFO_MAP)

def serialize_transform(t):
    from openravepy import quatFromRotationMatrix

    return {
        'position': list(map(float,t[0:3, 3])),
        'orientation': list(map(float,quatFromRotationMatrix(t[0:3, 0:3]))),
    }

# Deserialization.

class UnitaryMemoizer:
    """Memoizer which calls the given non-argument callable at most once.

    An instance of this class is initialized with a callable which
    takes no arguments.  When the instance is called the first time,
    it invokes the callable, saves the result, and returns it.
    Subsequent calls return the cached value.
    """
    def __init__(self, func):
        self.func = func
        self.called = False
    def __call__(self):
        if not self.called:
            self.result = self.func()
            self.called = True
        return self.result

def _deserialize_internal(env, data, data_type):
    from numpy import array, ndarray
    from openravepy import (Environment, KinBody, Robot, Trajectory,
                            RaveCreateTrajectory)
    from prpy.tsr import TSR, TSRChain
    from .exceptions import UnsupportedTypeDeserializationException

    if data_type == dict.__name__:
        return {
            deserialize(env, k): deserialize(env, v)
            for k, v in data.iteritems()
            if k != TYPE_KEY
        }
    elif data_type == ndarray.__name__:
        return array(data['data'])
    elif data_type in [ KinBody.__name__, Robot.__name__ ]:
        body = env.GetKinBody(data['name'])
        if body is None:
            raise ValueError('There is no body with name "{:s}".'.format(
                data['name']))

        return body
    elif data_type == KinBody.Link.__name__:
        body = env.GetKinBody(data['parent_name'])
        if body is None:
            raise ValueError('There is no body with name "{:s}".'.format(
                data['parent_name']))

        link = body.GetLink(data['name'])
        if link is None:
            raise ValueError('Body "{:s}" has no link named "{:s}".'.format(
                data['parent_name'], data['name']))

        return link
    elif data_type == KinBody.Joint.__name__:
        body = env.GetKinBody(data['parent_name'])
        if body is None:
            raise ValueError('There is no body with name "{:s}".'.format(
                data['parent_name']))

        joint = body.GetJoint(data['name'])
        if joint is None:
            raise ValueError('Body "{:s}" has no joint named "{:s}".'.format(
                data['parent_name'], data['name']))

        return joint
    elif data_type == Robot.Manipulator.__name__:
        body = env.GetKinBody(data['parent_name'])
        if body is None:
            raise ValueError('There is no robot with name "{:s}".'.format(
                data['parent_name']))
        elif not body.IsRobot():
            raise ValueError('Body "{:s}" is not a robot.'.format(
                data['parent_name']))

        manip = body.GetJoint(data['name'])
        if manip is None:
            raise ValueError('Robot "{:s}" has no manipulator named "{:s}".'.format(
                data['parent_name'], data['name']))

        return manip
    elif data_type == Trajectory.__name__:
        traj = RaveCreateTrajectory(env, '')
        traj.deserialize(data['data'])
        return traj
    elif data_type == TSR.__name__:
        return TSR.from_dict(data['data'])
    elif data_type == TSRChain.__name__:
        return TSRChain.from_dict(data['data'])
    else:
        raise UnsupportedTypeDeserializationException(data_type)

def deserialize(env, data):
    if isinstance(data, unicode):
        return data.encode()
    elif isinstance(data, list):
        return [ deserialize(env, x) for x in data ]
    elif isinstance(data, dict):
        return _deserialize_internal(env, data, data.get(TYPE_KEY))
    else:
        return data

def deserialize_environment(data, env=None, purge=False, reuse_bodies=None):
    import openravepy

    if env is None:
        env = openravepy.Environment()

    if reuse_bodies is None:
        reuse_bodies = []
    reuse_bodies_dict = { body.GetName(): body for body in reuse_bodies }
    reuse_bodies_set = set(reuse_bodies)

    # Release anything that's grabbed.
    for body in reuse_bodies:
        body.ReleaseAllGrabbed()

    # Remove any extra bodies from the environment.
    for body in env.GetBodies():
        if body not in reuse_bodies_set:
            deserialization_logger.debug('Purging body "%s".', body.GetName())
            env.Remove(body)

    # Create a or_ordf module on demand
    urdf_module_getter = UnitaryMemoizer(lambda: openravepy.RaveCreateModule(env,'urdf'))

    # Deserialize the kinematic structure.
    deserialized_bodies = []
    for body_data in data['bodies']:
        body = reuse_bodies_dict.get(body_data['name'], None)
        if body is None:
            body = deserialize_kinbody(env, body_data, state=False,
                urdf_module_getter=urdf_module_getter)

        deserialization_logger.debug('Deserialized body "%s".', body.GetName())
        deserialized_bodies.append((body, body_data))

    # Restore state. We do this in a second pass to insure that any bodies that
    # are grabbed already exist.
    for body, body_data in deserialized_bodies:
        deserialize_kinbody_state(body, body_data['kinbody_state'])

        if body.IsRobot():
            deserialize_robot_state(body, body_data['robot_state'])

    return env

def deserialize_kinbody(env, data, name=None, anonymous=False, state=True,
        urdf_module_getter=None):

    from openravepy import RaveCreateKinBody, RaveCreateRobot

    if urdf_module_getter is None:
        urdf_module_getter = UnitaryMemoizer(lambda: openravepy.RaveCreateModule(env,'urdf'))

    deserialization_logger.debug('Deserializing %s "%s".',
        'Robot' if data['is_robot'] else 'KinBody',
        data['name']
    )
    
    name_desired = name or data['name']
    
    if data.get('uri_only', False):
        
        if data['is_robot']:

            parts = data['uri'].split()
            if len(parts)==2 and parts[0].endswith('.urdf') and parts[1].endswith('.srdf'):
                module_urdf = urdf_module_getter()
                if module_urdf is None:
                    raise UnsupportedTypeDeserializationException('urdf srdf')
                robot_name = module_urdf.SendCommand('Load {}'.format(data['uri']))
                kinbody = env.GetRobot(robot_name)
                if robot_name != name_desired:
                    env.Remove(kinbody)
                    kinbody.SetName(name_desired)
                    env.Add(kinbody, anonymous)
            else:
                kinbody = env.ReadRobotXMLFile(data['uri'])
                kinbody.SetName(name_desired)
                env.Add(kinbody, anonymous)
            
        else:

            if data['uri'].endswith('.urdf'):
                module_urdf = urdf_module_getter()
                if module_urdf is None:
                    raise UnsupportedTypeDeserializationException('urdf')
                kinbody_name = module_urdf.SendCommand('Load {}'.format(data['uri']))
                kinbody = env.GetKinBody(kinbody_name)
                if kinbody_name != name_desired:
                    env.Remove(kinbody)
                    kinbody.SetName(name_desired)
                    env.Add(kinbody)
            else:
                kinbody = env.ReadKinBodyXMLFile(data['uri'])
                kinbody.SetName(name_desired)
                env.Add(kinbody)
        
    else:
        
        link_infos = [
            deserialize_link_info(link_data['info']) \
            for link_data in data['links']
        ]
        joint_infos = [
            deserialize_joint_info(joint_data['info']) \
            for joint_data in data['joints']
        ]

        if data['is_robot']:
            
            # TODO: Also load sensors.
            manipulator_infos = [
                deserialize_manipulator_info(manipulator_data['info']) \
                for manipulator_data in data['manipulators']
            ]
            sensor_infos = []
        
            kinbody = RaveCreateRobot(env, '')
            kinbody.Init(
                link_infos, joint_infos,
                manipulator_infos, sensor_infos,
                data['uri']
            )
        
        else:
            
            kinbody = RaveCreateKinBody(env, '')
            kinbody.Init(link_infos, joint_infos, data['uri'])
        
        kinbody.SetName(name_desired)
        env.Add(kinbody, anonymous)
    
    if state:
        deserialize_kinbody_state(kinbody, data['kinbody_state'])
        if kinbody.IsRobot():
            deserialize_robot_state(kinbody, data['robot_state'])

    return kinbody

def deserialize_kinbody_state(body, data):
    from openravepy import KinBody

    deserialization_logger.debug('Deserializing "%s" KinBody state.',
        body.GetName())

    for key, (_, set_fn) in KINBODY_STATE_MAP.iteritems():
        try:
            set_fn(body, data[key])
        except Exception as e:
            deserialization_logger.error(
                'Failed deserializing KinBody "%s" state "%s": %s',
                body.GetName(), key, e.message
            )
            raise

    link_transforms = data.get('link_transforms')
    dof_branches = data.get('dof_branches')
    if link_transforms is not None and dof_branches is not None:
        body.SetLinkTransformations(
            map(deserialize_transform, link_transforms),
            dof_branches
        )
    else:
        deserialization_logger.warn(
            'KinBody "{}" does not have link_transforms/dof_branches'
            ' saved; falling back to dof_values'.format(body.GetName()))
        body.SetDOFValues(data['dof_values'])

def deserialize_robot_state(body, data):
    deserialization_logger.debug('Deserializing "%s" Robot state.',
        body.GetName())

    for key, (_, set_fn) in ROBOT_STATE_MAP.iteritems():
        set_fn(body, data[key])

    env = body.GetEnv()

    for grabbed_info_dict in data['grabbed_bodies']:
        grabbed_info = deserialize_grabbed_info(grabbed_info_dict)

        robot_link = body.GetLink(grabbed_info._robotlinkname)
        robot_links_to_ignore = grabbed_info._setRobotLinksToIgnore

        grabbed_body = env.GetKinBody(grabbed_info._grabbedname)
        grabbed_pose = numpy.dot(robot_link.GetTransform(),
                                 grabbed_info._trelative)
        grabbed_body.SetTransform(grabbed_pose)

        body.Grab(grabbed_body, robot_link, robot_links_to_ignore)

def deserialize_with_map(obj, data, attribute_map):
    for key, (_, deserialize_fn) in attribute_map.iteritems():
        setattr(obj, key, deserialize_fn(data[key]))

    return obj

def deserialize_link_info(data):
    from openravepy import KinBody

    return deserialize_with_map(KinBody.LinkInfo(), data, LINK_INFO_MAP)
    
def deserialize_joint_info(data):
    from openravepy import KinBody

    return deserialize_with_map(KinBody.JointInfo(), data, JOINT_INFO_MAP)

def deserialize_manipulator_info(data):
    from openravepy import Robot

    return deserialize_with_map(Robot.ManipulatorInfo(), data, MANIPULATOR_INFO_MAP)

def deserialize_geometry_info(data):
    from openravepy import KinBody 

    geom_info = deserialize_with_map(
        KinBody.GeometryInfo(), data, GEOMETRY_INFO_MAP)

    # OpenRAVE only has a ReadTrimeshURI method on Environment. We create a
    # static, dummy environment (mesh_environment) just to load meshes.
    if geom_info._filenamecollision:
        geom_info._meshcollision = mesh_environment.ReadTrimeshURI(
            geom_info._filenamecollision)

    return geom_info

def deserialize_grabbed_info(data):
    from openravepy import Robot

    return deserialize_with_map(Robot.GrabbedInfo(), data, GRABBED_INFO_MAP)

def deserialize_transform(data):
    from openravepy import matrixFromQuat

    t = matrixFromQuat(data['orientation'])
    t[0:3, 3] = data['position']
    return t

# Schema.
mesh_environment = openravepy.Environment()
identity = lambda x: x
str_identity = (
    lambda x: x,
    lambda x: x.encode()
)
both_identity = (
    lambda x: x,
    lambda x: x
)
numpy_identity = (
    lambda x: x.tolist(),
    lambda x: numpy.array(x)
)
transform_identity = (
    serialize_transform,
    deserialize_transform
)


KINBODY_STATE_MAP = {
    'description': (
        lambda x: x.GetDescription(),
        lambda x, value: x.SetDescription(value),
    ),
    'link_enable_states': (
        lambda x: x.GetLinkEnableStates().tolist(),
        lambda x, value: x.SetLinkEnableStates(value)
    ),
    'link_velocities': (
        lambda x: x.GetLinkVelocities().tolist(),
        lambda x, value: x.SetLinkVelocities(value),
    ),
    'transform': (
        lambda x: serialize_transform(x.GetTransform()),
        lambda x, value: x.SetTransform(deserialize_transform(value)),
    ),
    'dof_weights': (
        lambda x: x.GetDOFWeights().tolist(),
        lambda x, value: x.SetDOFWeights(value),
    ),
    'dof_resolutions': (
        lambda x: x.GetDOFResolutions().tolist(),
        lambda x, value: x.SetDOFResolutions(value),
    ),
    'dof_position_limits': (
        lambda x: [ limits.tolist() for limits in x.GetDOFLimits() ],
        lambda x, (lower, upper): x.SetDOFLimits(lower, upper),
    ),
    'dof_velocity_limits': (
        lambda x: x.GetDOFVelocityLimits().tolist(),
        lambda x, value: x.SetDOFVelocityLimits(value),
    ),
    'dof_acceleration_limits': (
        lambda x: x.GetDOFAccelerationLimits().tolist(),
        lambda x, value: x.SetDOFAccelerationLimits(value),
    ),
    'dof_torque_limits': (
        lambda x: x.GetDOFTorqueLimits().tolist(),
        lambda x, value: x.SetDOFTorqueLimits(value),
    ),
    # TODO: What about link accelerations and geometry groups?
}
ROBOT_STATE_MAP = {
    # TODO: Does this preserve affine DOFs?
    'active_dof_indices': (
        lambda x: x.GetActiveDOFIndices().tolist(),
        lambda x, value: x.SetActiveDOFs(value)
    ),
    'active_manipulator': (
        lambda x: x.GetActiveManipulator().GetName(),
        lambda x, value: x.SetActiveManipulator(value),
    ),
}
LINK_INFO_MAP = {
    '_bIsEnabled': both_identity,
    '_bStatic': both_identity,
    '_mapFloatParameters': both_identity,
    '_mapIntParameters': both_identity,
    '_mapStringParameters': both_identity, # TODO
    '_mass': both_identity,
    '_name': str_identity,
    '_t': transform_identity,
    '_tMassFrame': transform_identity,
    '_vForcedAdjacentLinks': both_identity,
    '_vgeometryinfos': (
        lambda x: map(serialize_geometry_info, x),
        lambda x: map(deserialize_geometry_info, x),
    ),
    '_vinertiamoments': numpy_identity,
}
JOINT_INFO_MAP = {
    '_bIsActive': both_identity,
    '_bIsCircular': both_identity,
    '_linkname0':  str_identity,
    '_linkname1': str_identity,
    '_mapFloatParameters': both_identity,
    '_mapIntParameters': both_identity,
    '_mapStringParameters': both_identity, # TODO
    '_name': str_identity,
    '_type': (
        lambda x: x.name,
        lambda x: openravepy.KinBody.JointType.names[x].encode()
    ),
    '_vanchor': numpy_identity,
    '_vaxes': (
        lambda x: [ xi.tolist() for xi in x ],
        lambda x: map(numpy.array, x)
    ),
    '_vcurrentvalues': numpy_identity,
    '_vhardmaxvel': numpy_identity,
    '_vlowerlimit': numpy_identity,
    '_vmaxaccel': numpy_identity,
    '_vmaxinertia': numpy_identity,
    '_vmaxtorque': numpy_identity,
    '_vmaxvel': numpy_identity,
    '_vmimic': both_identity,
    '_voffsets': numpy_identity,
    '_vresolution': numpy_identity,
    '_vupperlimit': numpy_identity,
    '_vweights': numpy_identity,
}
GEOMETRY_INFO_MAP = {
    '_bModifiable': both_identity,
    '_bVisible': both_identity,
    '_fTransparency': both_identity,
    '_filenamecollision': str_identity,
    '_filenamerender': str_identity,
    '_t': transform_identity,
    '_type': (
        lambda x: x.name,
        lambda x: openravepy.GeometryType.names[x]
    ),
    '_vAmbientColor': numpy_identity,
    '_vCollisionScale': numpy_identity,
    '_vDiffuseColor': numpy_identity,
    '_vGeomData': numpy_identity,
    '_vRenderScale': numpy_identity,
    # TODO: What are these?
    #'_mapExtraGeometries': None
    #15 is not JSON serializable
    #'_trajfollow': None,
}
MANIPULATOR_INFO_MAP = {
    '_name': str_identity,
    '_sBaseLinkName': str_identity,
    '_sEffectorLinkName': str_identity,
    '_sIkSolverXMLId': str_identity,
    '_tLocalTool': transform_identity,
    '_vChuckingDirection': numpy_identity,
    '_vClosingDirection': numpy_identity,
    '_vGripperJointNames': both_identity, # TODO
    '_vdirection': numpy_identity,
}
GRABBED_INFO_MAP = {
    '_grabbedname': str_identity,
    '_robotlinkname': str_identity,
    '_setRobotLinksToIgnore': both_identity, # TODO
    '_trelative': transform_identity,
}
