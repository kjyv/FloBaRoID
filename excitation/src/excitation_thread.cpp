#include <yarp/os/all.h>

#include "excitation_thread.h"
#include "excitation_constants.h"

excitation_thread::excitation_thread(   std::string module_prefix,
                                        yarp::os::ResourceFinder rf,
                                        std::shared_ptr< paramHelp::ParamHelperServer > ph) :
                                            left_arm_chain_interface( "left_arm", module_prefix, get_robot_name() ),
                                            num_joints( left_arm_chain_interface.getNumberOfJoints() ),
                                            ref_speed_vector( num_joints ),
                                            command_interface( module_prefix ),
                                            generic_thread( module_prefix, rf, ph )
{
    // position mode on left arm chain interface
    left_arm_chain_interface.setPositionMode();
}

void excitation_thread::link_tutorial_params()
{
    // get a shared pointer to param helper
    std::shared_ptr<paramHelp::ParamHelperServer> ph = get_param_helper();
    // link the left_arm configuration (vector linking)
    // link the max_vel parameter (single value linking
    ph->linkParam( PARAM_ID_MAX_VEL, &max_vel );
}

bool excitation_thread::custom_init()
{
    // link the tutorial additional params to param helper
    link_tutorial_params();

    return true;
}

void excitation_thread::run()
{
    excitation_cmd.command = "";
    command_interface.getCommand(excitation_cmd, seq_num);

    if( excitation_cmd.command == "set_left_arm" ) {
        // set the ref speed for all the joints
        left_arm_chain_interface.setReferenceSpeed( max_vel );

        // position move to desired configuration
        yarp::sig::Vector pos;
        pos.resize(7); pos.zero();
        pos[0] = excitation_cmd.angle0;
        pos[1] = excitation_cmd.angle1;
        pos[2] = excitation_cmd.angle2;
        pos[3] = excitation_cmd.angle3;
        pos[4] = excitation_cmd.angle4;
        pos[5] = excitation_cmd.angle5;
        pos[6] = excitation_cmd.angle6;
        left_arm_chain_interface.move(pos);
    }
    else if( excitation_cmd.command != "" ) {
      std::cout << excitation_cmd.command <<  " -> command not valid" << std::endl;
    }
}

bool excitation_thread::custom_pause()
{
    // set the ref speed to 0 for all the joints
    left_arm_chain_interface.setReferenceSpeed( 0 );
}

bool excitation_thread::custom_resume()
{
    // set the ref speed to max_vel for all the joints
    left_arm_chain_interface.setReferenceSpeed( max_vel );
}
