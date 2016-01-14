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
     // left_arm chain interface
    walkman::yarp_single_chain_interface left_arm_chain_interface;
    // joints number
    int num_joints;
    // left_arm configuration vector
    yarp::sig::Vector left_arm_configuration;
    // max velocity
    double max_vel;
    // ref speed vector
    yarp::sig::Vector ref_speed_vector;
    // command interface
    walkman::yarp_custom_command_interface<excitation_msg> command_interface;
    excitation_msg excitation_cmd;
    int seq_num;

    // link the tutorial optional params
    void link_tutorial_params();

public:

    /**
     * @brief constructor
     *
     * @param module_prefix the prefix of the module
     * @param rf resource finderce
     * @param ph param helper
     */
     excitation_thread( std::string module_prefix, yarp::os::ResourceFinder rf, std::shared_ptr<paramHelp::ParamHelperServer> ph );

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

    /**
     * @brief custom_pause we use this method when a "pause" is sent to the module
     * @return true on success, false otherwise
     */
    virtual bool custom_pause();

    /**
     * @brief custom_resume we use this method when a "resume" is sent to the module
     * @return true on success, false otherwise
     */
    virtual bool custom_resume();
};

#endif
