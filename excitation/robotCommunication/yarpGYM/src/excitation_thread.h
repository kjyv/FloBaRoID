#ifndef excitation_THREAD_H_
#define excitation_THREAD_H_

#include <GYM/generic_thread.hpp>

#include <yarp/os/RateThread.h>
#include <yarp/sig/Vector.h>

#include <idynutils/yarp_single_chain_interface.h>
#include <GYM/yarp_command_interface.hpp>

#include "excitation_msg.h"

/**
 * @brief excitation control thread
 *
 **/
class excitation_thread : public generic_thread
{
private:
     // walkman chain interfaces
    walkman::yarp_single_chain_interface *right_leg_chain_if;
    walkman::yarp_single_chain_interface *left_leg_chain_if;
    // chain configuration vector
    yarp::sig::Vector chain_configuration;
    // joints number
    int num_joints;
    // ref speed vector
    yarp::sig::Vector ref_speed_vector;
    // command interface
    walkman::yarp_custom_command_interface<excitation_msg> command_interface;
    excitation_msg excitation_cmd;

    //data output
    yarp::os::Bottle outgoingData;
    yarp::os::Port outgoingPort;

public:

    /**
     * @brief constructor
     *
     * @param module_prefix the prefix of the module
     * @param rf resource finderce
     * @param ph param helper
     */
     excitation_thread( std::string module_prefix, yarp::os::ResourceFinder rf, std::shared_ptr<paramHelp::ParamHelperServer> ph );
     ~excitation_thread();

    /**
     * @brief excitation control thread initialization
     *
     * @return true on succes, false otherwise
     */
    virtual bool custom_init();

    /**
     * @brief excitation control thread main loop
     *
     */
    virtual void run();
};

#endif
