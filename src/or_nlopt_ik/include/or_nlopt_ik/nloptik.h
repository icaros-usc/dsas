#ifndef  NLOPTIK_H
#define  NLOPTIK_H

#include <nlopt.h>
#include <openrave/openrave.h>


class NloptIK : public OpenRAVE::IkSolverBase
{
  public:
    NloptIK(OpenRAVE::EnvironmentBasePtr penv);
    virtual bool Init(OpenRAVE::RobotBase::ManipulatorConstPtr pmanip);
    virtual OpenRAVE::RobotBase::ManipulatorPtr GetManipulator() const;
    virtual int GetNumFreeParameters() const;
    virtual bool GetFreeParameters(std::vector<double>&) const;
    virtual bool Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >);
    virtual bool Solve(const OpenRAVE::IkParameterization&, const std::vector<double>&, const std::vector<double>&, int, boost::shared_ptr<std::vector<double> >);
    virtual bool SolveAll(const OpenRAVE::IkParameterization&, int, std::vector<std::vector<double> >&);
    virtual bool SolveAll(const OpenRAVE::IkParameterization&, const std::vector<double>&, int, std::vector<std::vector<double> >&);
    virtual bool SetDistErrValue(std::ostream& sout, std::istream& sin);
    virtual bool SetAngleErrValue(std::ostream& sout, std::istream& sin);

    void wrapState(std::vector<double>& state);

    //nlopt variables
    std::vector<double> _target;
    OpenRAVE::RobotBasePtr _pRobot;
    OpenRAVE::RobotBase::ManipulatorPtr  _pmanip;
    OpenRAVE::RaveVector<double> quat_target;
    int numdofs;
    std::vector<int> _indices;
  private:
    bool SetTolValue(std::ostream& sout, std::istream& sin);
    bool GetSolutionError(const double *x, void *my_func_data, double &dist_err, double &angle_err);
    bool _initialized;

    //nlopt stuff
    nlopt_opt _opt;
    double _tol;
    double _dist_err_thres;
    double _angle_err_thres;
};


#endif

