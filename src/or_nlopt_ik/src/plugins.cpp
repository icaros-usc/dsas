#include <openrave/plugin.h>
#include <or_nlopt_ik/nloptik.h>

OpenRAVE::InterfaceBasePtr CreateInterfaceValidated(OpenRAVE::InterfaceType type, const std::string& interfacename, std::istream& sinput, OpenRAVE::EnvironmentBasePtr penv)
{
    if (type == OpenRAVE::PT_InverseKinematicsSolver && interfacename == "nloptik")
    {
        return OpenRAVE::InterfaceBasePtr(new NloptIK(penv));
    }
    return OpenRAVE::InterfaceBasePtr();
}

void GetPluginAttributesValidated(OpenRAVE::PLUGININFO& info)
{
    info.interfacenames[OpenRAVE::PT_InverseKinematicsSolver].push_back("NloptIK");
}

RAVE_PLUGIN_API void DestroyPlugin()
{
    RAVELOG_INFO("destroying plugin\n");
}
