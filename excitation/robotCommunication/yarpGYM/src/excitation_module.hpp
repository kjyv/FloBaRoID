#ifndef excitation_MODULE_HPP_
#define excitation_MODULE_HPP_

#include <GYM/generic_module.hpp>

#include "excitation_thread.h"
#include "excitation_constants.h"

/**
 * @brief excitation module derived from generic_module
 *
 * @author
 */
class excitation_module : public generic_module<excitation_thread> {
public:

    /**
     * @brief constructor: do nothing but construct the superclass
     *
     */
    excitation_module(int argc,
                      char* argv[],
                      std::string module_prefix,
                      int module_period,
                      yarp::os::ResourceFinder rf ) : generic_module<excitation_thread>( argc,
                                                                                         argv,
                                                                                         module_prefix,
                                                                                         module_period,
                                                                                         rf )
    {
    }

   /**
     * @brief custom_get_ph_parameters inherit from generic module, we reimplement it since we have more parameters in the
     * param_help (tutorial_configuration.ini file) than the default ones.
     * @return a vector of the custom parameters for the param helper
     */
    virtual std::vector< paramHelp::ParamProxyInterface* > custom_get_ph_parameters()
    {
        // custom param helper parameters vector
        std::vector<paramHelp::ParamProxyInterface *> custom_params;
        // insert max_vel param
        /*custom_params.push_back( new paramHelp::ParamProxyBasic<double>( "max_vel",
                                                                         PARAM_ID_MAX_VEL,
                                                                         PARAM_SIZE_MAX_VEL,
                                                                         paramHelp::PARAM_IN_OUT,
                                                                         NULL,
                                                                         "maximum velocity in [degree/second]" ) );
        */
        return custom_params;
    }
};

#endif
