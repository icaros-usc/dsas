#include <or_trac_ik/or_trac_ik.hpp>
#include <limits>       // std::numeric_limits
#include <iostream>

#include <ros/console.h>

//#include <ros/ros.h>
//#include <kdl/frames_io.hpp>
//#include <kdl/kinfam_io.hpp>


typedef OpenRAVE::RobotBase::RobotStateSaver RobotStateSaver;

namespace or_helper
{
    bool HasFlag(int key, OpenRAVE::IkFilterOptions options)
    {
        return (key & (static_cast<int>(options))) != 0;
    }

    bool IsSupported(int key)
    {
        return key == 0x0 ||
               key == OpenRAVE::IKFO_CheckEnvCollisions ||
               key == OpenRAVE::IKFO_IgnoreSelfCollisions ||
               key == (OpenRAVE::IKFO_CheckEnvCollisions | OpenRAVE::IKFO_IgnoreSelfCollisions);
    }
}


//initialize the kdl chain from openrave manip
void TracIK::InitKDLChain()
{
    _kdl_chain = KDL::Chain();

    //for each dof, add a joint to the kdl chain
    for (int i = 0; i < _numdofs; i++)
    {
        OpenRAVE::KinBody::JointPtr p_joint = _pRobot->GetJointFromDOFIndex(_indices[i]);
        //ROS_INFO_STREAM("segment from joint: " << getSegmentTransformFromJoint(p_joint));
        //_kdl_chain.addSegment(KDL::Segment(toKDLJoint(p_joint)));//, getSegmentTransformFromJoint(p_joint)));
        _kdl_chain.addSegment(KDL::Segment(toKDLJoint(p_joint), getSegmentTransformFromJoint(p_joint)));
    }

    //there may be a fixed transform between the end effector and last link
    _ee_to_last_joint = _pmanip->GetEndEffectorTransform().inverse() * _pRobot->GetJointFromDOFIndex(_indices.size()-1)->GetHierarchyChildLink()->GetTransform();

//    //TEST CHAIN
//    // PRINT TO MAKE SURE WE GET THE SAME AS READING URDF
//    std::vector<KDL::Segment> chain_segs = _kdl_chain.segments;
//    for(unsigned int i = 0; i < chain_segs.size(); ++i) {
//      ROS_INFO_STREAM("kdl joint origin: " << chain_segs[i].getJoint().JointOrigin() << "  axis: " << chain_segs[i].getJoint().JointAxis());
//      ROS_INFO_STREAM("kdl segment frame\n" << chain_segs[i].getFrameToTip());
//    }
//
//
//    KDL::ChainFkSolverPos_recursive fksolver = KDL::ChainFkSolverPos_recursive(_kdl_chain);
// 
//    // Create joint array
//    unsigned int nj = _kdl_chain.getNrOfJoints();
//    KDL::JntArray jointpositions = KDL::JntArray(nj);
// 
//    // Assign some values to the joint positions
//    // these are based on ada's home config
//    jointpositions(0) = -1.65549603;
//    jointpositions(1) = -1.48096311;
//    jointpositions(2) = 0.19731201;
//    jointpositions(3) = -1.10550746;
//    jointpositions(4) = 1.67789602;
//    jointpositions(5) = 3.39982207;
// 
//    // Create the frame that will contain the results
//    KDL::Frame cartpos;    
// 
//    // Calculate forward position kinematics
//    bool kinematics_status;
//    kinematics_status = fksolver.JntToCart(jointpositions,cartpos);
//    if(kinematics_status>=0){
//        ROS_INFO_STREAM("cartesian pose:\n" << cartpos);
//    }else{
//        ROS_INFO_STREAM("Error: could not calculate forward kinematics");
//    } 

}


KDL::Vector toKDLVec3(const OpenRAVE::Vector& vec)
{
    return KDL::Vector(vec.x, vec.y, vec.z);
}


KDL::Rotation toKDLQuat(const OpenRAVE::Vector& vec)
{
    return KDL::Rotation::Quaternion(vec[1], vec[2], vec[3], vec[0]);
}

KDL::Frame toKDLFrame(const OpenRAVE::Transform& transform)
{
    return KDL::Frame(toKDLQuat(transform.rot), toKDLVec3(transform.trans));
}


KDL::Joint toKDLJoint(const OpenRAVE::KinBody::JointPtr p_joint)
{
    KDL::Joint::JointType kdl_type;

    //figure out joint type
    OpenRAVE::KinBody::JointType or_j_type = p_joint->GetType();
    //TODO: handle other joint types?
    if (or_j_type == OpenRAVE::KinBody::JointRevolute)
    {
        kdl_type = KDL::Joint::RotAxis;
    } else if (or_j_type == OpenRAVE::KinBody::JointSlider || or_j_type == OpenRAVE::KinBody::JointPrismatic) {
        kdl_type = KDL::Joint::TransAxis;
    } else {
        //ROS_FATAL_STREAM("Error: Unknown conversion to kdl for joint type " << or_j_type);
        RAVELOG_ERROR("Error: Unknown conversion to kdl for joint type %s", or_j_type);
    }

    //get origin and axis from parent joint
    KDL::Vector origin = toKDLVec3(p_joint->GetInfo()._vanchor);
    KDL::Vector axis = toKDLVec3(p_joint->GetInfo()._vaxes[0]);  //TODO can we always just take 0? urdf_loader only sets that one

    return KDL::Joint(p_joint->GetName(), origin, axis, kdl_type);
}

KDL::Frame getSegmentTransformFromJoint(const OpenRAVE::KinBody::JointPtr p_joint)
{
    return toKDLFrame(p_joint->GetInternalHierarchyLeftTransform() * p_joint->GetInternalHierarchyRightTransform());
}


void TracIK::InitKDLJointLimits()
{
    _l_limits.resize(_numdofs);
    _u_limits.resize(_numdofs);
    _is_continous_joint.resize(_numdofs);
    for (int i = 0; i < _numdofs; i++)
    {
        //if(limits[ii].first<-2*M_PI){
        OpenRAVE::KinBody::JointPtr p_joint = _pRobot->GetJointFromDOFIndex(_indices[i]);
        if (p_joint->IsCircular(0))
        {
            _l_limits(i)=std::numeric_limits<float>::lowest();
            _u_limits(i)=std::numeric_limits<float>::max();
            _is_continous_joint[i] = true;
        } else {
            _l_limits(i) = p_joint->GetLimit().first;
            _u_limits(i) = p_joint->GetLimit().second;
            _is_continous_joint[i] = false;
        }
    }
}


bool TracIK::Init(OpenRAVE::RobotBase::ManipulatorConstPtr pmanip)
{
    _pmanip = boost::const_pointer_cast<OpenRAVE::RobotBase::Manipulator>(pmanip);
    _pmanip_base = _pmanip->GetBase();
    _pRobot = _pmanip->GetRobot();
    _indices = _pmanip->GetArmIndices();
    _numdofs = _pmanip->GetArmDOF();

    InitKDLChain();
    //InitTracIKSolver();
    return true;
}


TracIK::TracIK(OpenRAVE::EnvironmentBasePtr penv) :
                IkSolverBase(penv)
{
    __description = ":Interface Author: Shervin Javdani";

    //_target.reserve(7);
    //_initialized = false;
}


TracIK::~TracIK()
{
//  delete _tracik_solver_;
}


OpenRAVE::RobotBase::ManipulatorPtr TracIK::GetManipulator() const
{
    return _pmanip;
}

int TracIK::GetNumFreeParameters() const
{
    return 0;
}

bool TracIK::GetFreeParameters(std::vector<double> &v) const
{
    v.clear();
    return true;
}

KDL::JntArray toKDLJntArray(const std::vector<double>& vec)
{
    KDL::JntArray to_ret(vec.size());
    for (size_t i=0; i < vec.size(); i++)
    {
        to_ret(i) = vec[i];
    }
    return to_ret;
}

std::vector<double> toStdVec(const KDL::JntArray& arr)
{
    std::vector<double> vec(arr.rows());
    toStdVec(arr, vec);
    return vec;
}

void toStdVec(const KDL::JntArray& arr, std::vector<double>& vec)
{
    vec.resize(arr.rows());
    for (size_t i=0; i < vec.size(); i++)
    {
        vec[i] = arr(i);
    }
}

bool TracIK::Solve(const OpenRAVE::IkParameterization& params, const std::vector<double>& q0, int filter_options, boost::shared_ptr<std::vector<double> > result)
{
    //reinitialize solver in case bounds changed
    InitKDLJointLimits();
    return Solve_NoInit(params, q0, filter_options, result);
}

bool TracIK::Solve_NoInit(const OpenRAVE::IkParameterization& params, const std::vector<double>& q0, int filter_options, boost::shared_ptr<std::vector<double> > result)
{
    //_ee_to_last_joint = _pmanip->GetEndEffectorTransform().inverse() * _pRobot->GetJointFromDOFIndex(_indices.size()-1)->GetHierarchyChildLink()->GetTransform();
    TRAC_IK::TRAC_IK tracik_solver(_kdl_chain, _l_limits, _u_limits);

    //target transform is transform between the base link and what is specified by params
    //KDL::Frame target_transform= toKDLFrame(_pmanip_base->GetTransform().inverse() * params.GetTransform6D());
    //actually that was buggy for some reason, I get the transform in params is already relative?
    KDL::Frame target_transform= toKDLFrame(params.GetTransform6D() * _ee_to_last_joint);
    KDL::JntArray tracik_result(q0.size());

    int tracik_return = tracik_solver.CartToJnt(toKDLJntArray(q0), target_transform, tracik_result);
    //if tracik said no solution, return now
    //
    if (tracik_return <= 0)
    {
        return false;
    }

    //std::vector<double> q0_copy(q0);
    toStdVec(tracik_result, *(result.get()));

//    for (int i = 0; i < _numdofs; i++)
//    {
//        //if(limits[ii].first<-2*M_PI){
//        OpenRAVE::KinBody::JointPtr p_joint = _pRobot->GetJointFromDOFIndex(_indices[i]);
//        if (p_joint->IsCircular(0))
//        {
//            double result_this_joint = result->at(i);
//            while ( std::abs(q0_copy[i] - (result_this_joint + 2.*M_PI)) < std::abs(q0_copy[i] - result_this_joint))
//            {
//                result_this_joint += 2.*M_PI;
//            }
//            while ( std::abs(q0_copy[i] - (result_this_joint - 2.*M_PI)) < std::abs(q0_copy[i] - result_this_joint))
//            {
//                result_this_joint -= 2.*M_PI;
//            }
//            result->at(i) = result_this_joint;
//        }
//    }

    bool checkSelfCollision = !or_helper::HasFlag(filter_options, OpenRAVE::IKFO_IgnoreSelfCollisions);
    bool checkEnvCollision = or_helper::HasFlag(filter_options, OpenRAVE::IKFO_CheckEnvCollisions);

    if (!or_helper::IsSupported(filter_options))
    {
        RAVELOG_WARN("Unsupported filter option %#x. Supported options are:"
                     " %#x, %#x, %#x and %#x.\n",
            filter_options,
            0,
              OpenRAVE::IKFO_CheckEnvCollisions,
              OpenRAVE::IKFO_IgnoreSelfCollisions,
              OpenRAVE::IKFO_CheckEnvCollisions
            | OpenRAVE::IKFO_IgnoreSelfCollisions
        );
    }

    if(checkSelfCollision || checkEnvCollision)
    {
        //RobotStateSaver const saver(_pRobot, OpenRAVE::KinBody::Save_ActiveDOF | OpenRAVE::KinBody::Save_LinkTransformation);
        RobotStateSaver const saver(_pRobot, OpenRAVE::KinBody::Save_ActiveDOF | OpenRAVE::KinBody::Save_LinkTransformation);
        _pRobot->SetDOFValues(*result.get(), true, _pmanip->GetArmIndices());

        if(checkSelfCollision && _pRobot->CheckSelfCollision())
        {
            RAVELOG_DEBUG("No IK solution, robot in self collision.\n");
            return false;
        }

        if(checkEnvCollision && GetEnv()->CheckCollision(_pRobot))
        {
            RAVELOG_DEBUG("Warning: no IK solution, robot colliding with environment.\n");
            return false;
        }
    }

    return true;
}

bool TracIK::Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >)
{
    RAVELOG_ERROR("Function bool Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >)not implemented in TracIK.\n");
    return false;
}

bool TracIK::SolveAll(const OpenRAVE::IkParameterization& param, int filterOptions, std::vector<std::vector<double> >& returnValues)
{
    std::vector<double> q0(_numdofs, 0.0);
    return SolveAll(param, q0, filterOptions, returnValues);
}

bool TracIK::SolveAll(const OpenRAVE::IkParameterization& param, const std::vector<double>& q0, int filterOptions, std::vector<std::vector<double> >& returnValues)
{
    //reinitialize solver in case bounds changed
    InitKDLJointLimits();

    std::vector<double> randSample(_numdofs, 0.0);
    std::vector<double>* solution = new std::vector<double>(_numdofs);
    boost::shared_ptr<std::vector<double> > solutionPtr(solution);

    // Try 1K random samples as starting points
    for (size_t randomSamples = 0; randomSamples < 1000; randomSamples++)
    {
        // Sample joint values
        for (int j = 0; j < _numdofs; j++)
        {
            // if continuous, sample between -pi and pi from current
            if (_is_continous_joint[j])
            {
              randSample[j] = 2.0 * (((double)(rand()) / (double)(RAND_MAX)) * M_PI - M_PI * 0.5) + q0[j];
            } else {
              // otherwise, sample between limits
              randSample[j] = ((double)(rand()) / (double)(RAND_MAX)) * (_u_limits(j)-_l_limits(j)) - _l_limits(j);
            }
        }

        if (Solve_NoInit(param, randSample, filterOptions, solutionPtr) )
        {
            returnValues.push_back(*solution);
        }

    }
    return !returnValues.empty();
}
