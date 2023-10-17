/** \file urdf_loader.h
 * \brief Interface to a URDF loading plugin for OpenRAVE
 * \author Pras Velagapudi
 * \date 2013
 */

/* (C) Copyright 2013 Carnegie Mellon University */
#ifndef URDF_LOADER_H
#define URDF_LOADER_H

#include <openrave/openrave.h>
#include <openrave/plugin.h>
#include <boost/bind.hpp>
#include <urdf/model.h>
#include <srdfdom/model.h>

#include <tinyxml.h>

namespace or_urdf
{
  class URDFLoader : public OpenRAVE::ModuleBase
  {
  public:
    /** Opens a URDF file and returns a robot in OpenRAVE */
    bool load(std::ostream& sout, std::istream& sin);
    
    /** Constructs plugin and registers functions */
    URDFLoader(OpenRAVE::EnvironmentBasePtr env) : OpenRAVE::ModuleBase(env)
    {
      __description = "URDFLoader: Loader that imports URDF files.";
      _env = env;

      RegisterCommand("load", boost::bind(&URDFLoader::load, this, _1, _2),
                      "load URDF and SRDF from file");
    }

    void Destroy() { RAVELOG_INFO("URDF loader unloaded from environment\n"); }
    
    virtual ~URDFLoader() {}

    void ParseURDF(urdf::Model &model, std::vector<OpenRAVE::KinBody::LinkInfoPtr> &link_infos,
                   std::vector<OpenRAVE::KinBody::JointInfoPtr> &joint_infos);

    void ParseSRDF(urdf::Model const &urdf,
                   srdf::Model const &srdf,
                   std::vector<OpenRAVE::KinBody::LinkInfoPtr> &link_infos,
                   std::vector<OpenRAVE::KinBody::JointInfoPtr> &joint_infos,
                   std::vector<OpenRAVE::RobotBase::ManipulatorInfoPtr> &manip_infos);

    void ProcessGeometryGroupTagsFromURDF(
                   TiXmlDocument &xml_doc,
                   std::vector<OpenRAVE::KinBody::LinkInfoPtr> &link_infos);
    
    /* This is called on env.LoadProblem(m, 'command') */
    int main(const std::string& cmd) { RAVELOG_INFO("URDF loader initialized with command: %s\n", cmd.c_str()); return 0; }

  private:
    /** Reference to OpenRAVE environment, filled in on construction */
    OpenRAVE::EnvironmentBasePtr _env;
  };
  
} /* namespace or_urdf */

#endif // URDF_LOADER_H
