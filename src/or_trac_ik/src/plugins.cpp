#include <openrave/plugin.h>
#include <or_trac_ik/or_trac_ik.hpp>

OpenRAVE::InterfaceBasePtr CreateInterfaceValidated(OpenRAVE::InterfaceType type, const std::string& interfacename, std::istream& sinput, OpenRAVE::EnvironmentBasePtr penv)
{
    if (type == OpenRAVE::PT_InverseKinematicsSolver && interfacename == "tracik")
    {
        return OpenRAVE::InterfaceBasePtr(new TracIK(penv));
    }
    return OpenRAVE::InterfaceBasePtr();
}

void GetPluginAttributesValidated(OpenRAVE::PLUGININFO& info)
{
    info.interfacenames[OpenRAVE::PT_InverseKinematicsSolver].push_back("TracIK");
}

RAVE_PLUGIN_API void DestroyPlugin()
{
    RAVELOG_INFO("destroying plugin\n");
}

