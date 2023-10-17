#ifndef  TRACIK_H
#define  TRACIK_H

#include <trac_ik/trac_ik.hpp>
#include <openrave/openrave.h>

#include <boost/shared_ptr.hpp>


KDL::Joint toKDLJoint(const OpenRAVE::KinBody::JointPtr p_joint);
KDL::Vector toKDLVec3(const OpenRAVE::Vector& vec);
KDL::Rotation toKDLQuat(const OpenRAVE::Vector& vec);
KDL::Frame toKDLFrame(const OpenRAVE::Transform& transform);

KDL::JntArray toKDLJntArray(const std::vector<double>& vec);
std::vector<double> toStdVec(const KDL::JntArray& arr);
void toStdVec(const KDL::JntArray& arr, std::vector<double>& vec);

KDL::Frame getSegmentTransformFromJoint(const OpenRAVE::KinBody::JointPtr p_joint);

class TracIK : public OpenRAVE::IkSolverBase
{
  public:
    TracIK(OpenRAVE::EnvironmentBasePtr penv);
    ~TracIK();

    virtual bool Init(OpenRAVE::RobotBase::ManipulatorConstPtr pmanip);
    virtual OpenRAVE::RobotBase::ManipulatorPtr GetManipulator() const;
    virtual int GetNumFreeParameters() const;
    virtual bool GetFreeParameters(std::vector<double>&) const;
    virtual bool Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >);
    bool Solve_NoInit(const OpenRAVE::IkParameterization&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >);
    virtual bool Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >);
    virtual bool SolveAll(const OpenRAVE::IkParameterization&, int, std::vector<std::vector<double> >&);
    virtual bool SolveAll(const OpenRAVE::IkParameterization&, const std::vector<double>&, int, std::vector<std::vector<double> >&);

    void InitKDLChain();
    void InitKDLJointLimits();

    //void wrapState(std::vector<double>& state);

    //nlopt variables
    //std::vector<double> _target;
    //OpenRAVE::RaveVector<double> quat_target;
  private:
    //bool GetSolutionError(const double *x, void *my_func_data, double &dist_err, double &angle_err);
//    bool _initialized;

    OpenRAVE::RobotBase::ManipulatorPtr  _pmanip;
    OpenRAVE::RobotBasePtr _pRobot;
    OpenRAVE::KinBody::LinkPtr  _pmanip_base;
    OpenRAVE::Transform _ee_to_last_joint;
    std::vector<int> _indices;
    int _numdofs;

    KDL::Chain _kdl_chain;
    std::vector<bool> _is_continous_joint;
    KDL::JntArray _l_limits, _u_limits;

    //TRAC_IK::TRAC_IK* _tracik_solver_ = NULL;
};



#endif


