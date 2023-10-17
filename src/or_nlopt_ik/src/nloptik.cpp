#include <math.h>
#include <or_nlopt_ik/nloptik.h>
#include <iostream>

typedef OpenRAVE::RobotBase::RobotStateSaver RobotStateSaver;
typedef std::pair<double, double> Pair;
typedef std::vector<Pair> v_Pairs;

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

static double obj_func(unsigned n, const double *x, double *grad, void *my_func_data)
{

    NloptIK *nlopt_p = (NloptIK *) my_func_data;

    std::vector<double> q(nlopt_p->numdofs);
    for (int ii = 0; ii < nlopt_p->numdofs; ii++)
    {
        q[ii] = x[ii];
    }
    nlopt_p->_pRobot->SetActiveDOFValues(q);
    OpenRAVE::Transform transform = nlopt_p->_pmanip->GetEndEffectorTransform();
    OpenRAVE::RaveVector<double> trans_v = transform.trans;
    OpenRAVE::RaveVector<double> rot_v = transform.rot;
    double quat_diff = rot_v.dot(nlopt_p->quat_target)  - 1.0;
    rot_v = rot_v.normalize4();
    double res = pow((trans_v[0] - nlopt_p->_target[0]), 2) +
                 pow((trans_v[1] - nlopt_p->_target[1]), 2) +
                 pow((trans_v[2] - nlopt_p->_target[2]), 2) +
                 pow(quat_diff, 2);

    //gradient stuff
    if (grad)
    {
        std::vector<double> jacobian;
        std::vector<double> rotationJacobian;

        nlopt_p->_pmanip->CalculateJacobian(jacobian);
        nlopt_p->_pmanip->CalculateRotationJacobian(rotationJacobian);

        OpenRAVE::RaveVector<double> rv_target;
        OpenRAVE::RaveVector<double> error;

        rv_target.Set3(nlopt_p->_target[0], nlopt_p->_target[1], nlopt_p->_target[2]);
        error = trans_v - rv_target;
        double quat_error = nlopt_p->quat_target.dot(rot_v) - 1;
        double trans_grad[6];
        double rot_grad[6];

        for (int ii = 0; ii < 6; ii++)
        {
            trans_grad[ii] = 2 * (error[0] * jacobian[ii] + error[1] * jacobian[nlopt_p->numdofs + ii] + error[2] * jacobian[2 * nlopt_p->numdofs + ii]);
            rot_grad[ii] = 2 * quat_error * (nlopt_p->quat_target[0] * rotationJacobian[ii] +
                                             nlopt_p->quat_target[1] * rotationJacobian[nlopt_p->numdofs + ii] +
                                             nlopt_p->quat_target[2] * rotationJacobian[2 * nlopt_p->numdofs + ii] +
                                             nlopt_p->quat_target[3] * rotationJacobian[3 * nlopt_p->numdofs + ii]);
            grad[ii] = trans_grad[ii] + rot_grad[ii];
        }

    }
    return res;
}

NloptIK::NloptIK(OpenRAVE::EnvironmentBasePtr penv) :
                IkSolverBase(penv)
{
    __description = ":Interface Author: Stefanos Nikolaidis";
    OpenRAVE::InterfaceBase::RegisterCommand("SetTolValue", boost::bind(&NloptIK::SetTolValue, this, _1, _2), "sets the tolerance value for constraints and objective function, default is: 1e-8");

    _target.reserve(7);
    _initialized = false;
}

bool NloptIK::SetDistErrValue(std::ostream& sout, std::istream& sin)
{
    std::string s_value;
    sin >> s_value;
    _dist_err_thres = std::stod(s_value);
    sout << "ok\n";
    return true;
}

bool NloptIK::SetAngleErrValue(std::ostream& sout, std::istream& sin)
{
    std::string s_value;
    sin >> s_value;
    _angle_err_thres = std::stod(s_value);
    sout << "ok\n";
    return true;
}

bool NloptIK::GetSolutionError(const double *x, void *my_func_data, double &dist_err, double &angle_err)
{
    NloptIK *nlopt_p = (NloptIK *) my_func_data;

    std::vector<double> q(nlopt_p->numdofs);
    for (int ii = 0; ii < nlopt_p->numdofs; ii++)
    {
        q[ii] = x[ii];
    }
    nlopt_p->_pRobot->SetActiveDOFValues(q);
    OpenRAVE::Transform transform = nlopt_p->_pmanip->GetEndEffectorTransform();
    OpenRAVE::RaveVector<double> trans_v = transform.trans;
    OpenRAVE::RaveVector<double> rot_v = transform.rot;

    rot_v = rot_v.normalize4();
    dist_err = sqrt(pow((trans_v[0] - nlopt_p->_target[0]), 2) + pow((trans_v[1] - nlopt_p->_target[1]), 2) + pow((trans_v[2] - nlopt_p->_target[2]), 2));

    OpenRAVE::RaveVector<double> quat_curr_wrt_targ = OpenRAVE::geometry::quatMultiply(OpenRAVE::geometry::quatInverse(quat_target), rot_v);
    quat_curr_wrt_targ = quat_curr_wrt_targ * (1.0 / quat_curr_wrt_targ.lengthsqr4());
    angle_err = 2 * acos(quat_curr_wrt_targ[0]);

    if (angle_err > M_PI)
    {
        angle_err = -2 * M_PI + angle_err;
    }
    if (angle_err < -M_PI)
    {
        angle_err = 2 * M_PI + angle_err;
    }
    angle_err = angle_err * 180.0 / M_PI;
    return true;
}

bool NloptIK::SetTolValue(std::ostream& sout, std::istream& sin)
{
    std::string s_value;
    sin >> s_value;
    _tol = std::stod(s_value);
    sout << "ok\n";
    return true;
}

bool NloptIK::Init(OpenRAVE::RobotBase::ManipulatorConstPtr pmanip)
{
    if (_initialized)
    {
        nlopt_destroy(_opt);
        _opt = NULL;
    }

    _pmanip = boost::const_pointer_cast<OpenRAVE::RobotBase::Manipulator>(pmanip);
    _pRobot = _pmanip->GetRobot();
    numdofs = _pmanip->GetArmDOF();        //_pRobot->GetActiveDOF();
    _indices = _pmanip->GetArmIndices();


    _opt = nlopt_create(NLOPT_LD_SLSQP, numdofs); /* algorithm and dimensionality */ //.. best so far 943 msec, 0.88 success rate
    nlopt_set_min_objective(_opt, obj_func, this);

    //get limits
    v_Pairs limits;
    for (int ii = 0; ii < numdofs; ii++)
    {
        OpenRAVE::KinBody::JointPtr p_joint = _pRobot->GetJointFromDOFIndex(_indices[ii]);
        Pair limit = p_joint->GetLimit();
        limits.push_back(limit);
    }

    double d_lowerLimit[numdofs];
    double d_upperLimit[numdofs];

    for (int ii = 0; ii < numdofs; ii++)
    {
        //if(limits[ii].first<-2*M_PI){
        OpenRAVE::KinBody::JointPtr p_joint = _pRobot->GetJointFromDOFIndex(_indices[ii]);
        if (p_joint->IsCircular(0))
        {
            d_lowerLimit[ii] = -HUGE_VAL;
        }
        else
        {
            d_lowerLimit[ii] = limits[ii].first;
        }
    }

    for (int ii = 0; ii < numdofs; ii++)
    {
        OpenRAVE::KinBody::JointPtr p_joint = _pRobot->GetJointFromDOFIndex(_indices[ii]);
        if (p_joint->IsCircular(0))
        {
            d_upperLimit[ii] = HUGE_VAL;
        }
        else
        {
            d_upperLimit[ii] = limits[ii].second;
        }
    }

    nlopt_set_lower_bounds(_opt, d_lowerLimit);
    nlopt_set_upper_bounds(_opt, d_upperLimit);

    nlopt_set_maxtime(_opt, 0.050);

    //get the resolution of joints and compute the minimum
    std::vector<double> d_resolution;
    _pRobot->GetActiveDOFResolutions(d_resolution);
    double min_resol = *std::min_element(d_resolution.begin(), d_resolution.end());

    //make tolerance of variables proportional to the resolution of joints
    //min_resol = min_resol / 1000.;
    //nlopt_set_xtol_rel(_opt, min_resol);
    //nlopt_set_xtol_rel(_opt,1e-8);

    //set square of minimum resolution for objective function
    nlopt_set_stopval(_opt, 1e-8);

    _dist_err_thres = 0.02;
    _angle_err_thres = 5;

    _initialized = true;
    return true;
}

OpenRAVE::RobotBase::ManipulatorPtr NloptIK::GetManipulator() const
{
    return _pmanip;
}

int NloptIK::GetNumFreeParameters() const
{
    return 0;
}

bool NloptIK::GetFreeParameters(std::vector<double> &v) const
{
    v.clear();
    return true;
}

void NloptIK::wrapState(std::vector<double>& state)
{
    for (size_t i = 0; i < this->_indices.size(); i++)
    {
         OpenRAVE::RobotBase::JointPtr joint = _pRobot->GetJoints()[this->_indices[i]];

         if (joint->IsCircular(0))
         {
             double v = fmod(state[i], 2.0 * M_PI);
             while (v <= -M_PI)
              v += 2.0 * M_PI;
             
             while (v > M_PI)
              	v -= 2.0 * M_PI;
             state[i] = v;
         }
    }
}

bool NloptIK::Solve(const OpenRAVE::IkParameterization& params, const std::vector<double>& q0, int filter_options, boost::shared_ptr<std::vector<double> > result)
{
    RobotStateSaver const saver(_pRobot, OpenRAVE::KinBody::Save_ActiveDOF | OpenRAVE::KinBody::Save_LinkTransformation);
    _pRobot->SetActiveDOFs(_pmanip->GetArmIndices());
    OpenRAVE::geometry::RaveVector<double> trans = params.GetTranslation3D();
    OpenRAVE::geometry::RaveVector<double> rot = params.GetRotation3D();

    //transform to origin
    OpenRAVE::geometry::RaveTransform<double> transform = _pRobot->GetTransform();
    transform.identity();
    _pRobot->SetTransform(transform);

    for (int ii = 0; ii < 3; ii++)
    {
        _target[ii] = trans[ii];
    }
    for (int ii = 3; ii < 7; ii++)
    {
        _target[ii] = rot[ii - 3];
    }
    quat_target =
    {   _target[3], _target[4], _target[5], _target[6]};
    //initialization
    double x[numdofs];

    for (int ii = 0; ii < numdofs; ii++)
    {
        x[ii] = q0[ii];
    }
    double minf;

    std::vector<double> jacobian;
    if (nlopt_optimize(_opt, x, &minf) < 0)
    {
        RAVELOG_DEBUG("NLOPT Optimization failed.\n");
        return false;
    }

    double dist_err, angle_err;
    GetSolutionError(x, this, dist_err, angle_err);
    if ((dist_err > _dist_err_thres) || (angle_err > _angle_err_thres))
    {
        RAVELOG_DEBUG("Error from solution returned by NLOPT Optimization larger than pre-specified threshold.\n");
        return false;
    }

    bool checkSelfCollision = !or_helper::HasFlag(filter_options, OpenRAVE::IKFO_IgnoreSelfCollisions);
    bool checkEnvCollision = or_helper::HasFlag(filter_options, OpenRAVE::IKFO_CheckEnvCollisions);


    std::vector<OpenRAVE::dReal> q_s(numdofs);
    for (int i = 0; i < numdofs; i++)
    {
        q_s[i] = x[i];
    }

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
        _pRobot->SetActiveDOFValues(q_s, true);

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
    wrapState(q_s);
    *result.get() = q_s;
    return true;
}

bool NloptIK::Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >)
{
    RAVELOG_ERROR("Function bool Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >)not implemented in NloptIK.\n");
    return false;
}

bool NloptIK::SolveAll(const OpenRAVE::IkParameterization& param, int filterOptions, std::vector<std::vector<double> >& returnValues)
{
    std::vector<double> randSample(numdofs, 0.0);
    std::vector<double>* solution = new std::vector<double>();
    for (int j = 0; j < numdofs; j++)
    {
        solution->push_back(0.0);
    }
    boost::shared_ptr<std::vector<double> > solutionPtr(solution);
    bool foundOne = false;
    // Try 1K random samples as starting points
    for (size_t randomSamples = 0; randomSamples < 1000; randomSamples++)
    {
        // Sample between -PI and PI
        for (int j = 0; j < numdofs; j++)
        {
            randSample[j] = 2.0 * (((double)(rand()) / (double)(RAND_MAX)) * M_PI - M_PI * 0.5);
        }

        for (int j = 0; j < numdofs; j++)
        {
            (*solution)[j] = 0.0;
        }
        // Solve for this random seed
        if (Solve(param, randSample, filterOptions, solutionPtr) )
        {
            bool isNear = false;
            // Check all the return values to see if we already got a solution like this one
            for (size_t s = 0; s < returnValues.size(); s++)
            {
                bool nearSolution = true;
                const std::vector<double>& currentSol = returnValues[s];
                for (int j = 0; j < numdofs; j++)
                {
                    double diff = fabs(currentSol[j] - (*solution)[j]);

                    if (diff > 1e-1)
                    {
                        nearSolution = false;
                        break;
                    }
                }

                if (nearSolution)
                {
                    isNear = true;
                    break;
                }
            }

            // If this is far enough away from other solutions, accept it.
            if (!isNear)
            {
                foundOne = true;
                returnValues.push_back(*solution);
            }
        }
    }

    return foundOne;
    //RAVELOG_ERROR("WARNING: Function bool SolveAll(const OpenRAVE::IkParameterization&, const std::vector<double>&, int, std::vector<std::vector<double> >&)not implemented in NloptIK.\n");
    //return false;
}

bool NloptIK::SolveAll(const OpenRAVE::IkParameterization& param, const std::vector<double>& q0, int filterOptions, std::vector<std::vector<double> >& returnValues)
{
    return SolveAll(param, filterOptions, returnValues);
}
