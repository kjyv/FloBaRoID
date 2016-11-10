#include <yarp/os/all.h>

#include "excitation_thread.h"
#include "excitation_constants.h"
#include "saveEigen.h"

#include <Eigen/Eigen>

#include <list>
#include <vector>

using namespace std;
using namespace Eigen;
list < VectorXd > tmpLog;
excitation_thread::excitation_thread(   std::string module_prefix,
                                        yarp::os::ResourceFinder rf,
                                        std::shared_ptr< paramHelp::ParamHelperServer > ph) :
                                            ref_speed_vector( num_joints ),
                                            command_interface( module_prefix ),
                                            generic_thread( module_prefix, rf, ph )
{
    right_leg_chain_if = new walkman::yarp_single_chain_interface("right_leg", module_prefix, get_robot_name());
    left_leg_chain_if = new walkman::yarp_single_chain_interface("left_leg", module_prefix, get_robot_name());
    num_joints = right_leg_chain_if->getNumberOfJoints() + left_leg_chain_if->getNumberOfJoints();

    // set controller mode on chain interface
    right_leg_chain_if->setPositionDirectMode();   //sets setpoint directly, uses maximum speed all the time
    left_leg_chain_if->setPositionDirectMode();
    //chain_interface->setPositionMode();   //generate trajectories, allows setting velocities
    //chain_interface->setVelocityMode();

    //init velocity control with zero velocity
    /*yarp::sig::Vector vels;
    if(vels.size()!=num_joints) vels.resize(num_joints);
    vels.zero();
    chain_interface->move(vels);*/

    //open data output port
    outgoingPort.open("/" + module_prefix + "/state:o");
}

excitation_thread::~excitation_thread()
{
    //save(&tmpLog, "log.csv");
}

bool excitation_thread::custom_init()
{
    // get a shared pointer to param helper
    std::shared_ptr<paramHelp::ParamHelperServer> ph = get_param_helper();
    // link parameters (single value linking)
    //ph->linkParam( PARAM_ID_N_DOFS, &n_dofs );

    return true;
}

void excitation_thread::run()
{
    static yarp::sig::Vector pos; pos.resize(num_joints);
    //static yarp::sig::Vector vels;
    //if(vels.size()!=num_joints) vels.resize(num_joints);

    static yarp::sig::Vector q_r, q_l, qdot_r, qdot_l, tau_r, tau_l;
    double t = yarp::os::Time::now();

    /*
    double l=1, w_f=1, a=0.5, b=0.8, q0=-35, nf=1;
    double angle = (a/(w_f*l)*sin(w_f*l*t) +  a/(w_f*l)*cos(w_f*l*t))/M_PI*180;

    if(pos.size()!=num_joints) pos.resize(num_joints);
    pos.zero();
    /*pos[0] = excitation_cmd.angle0;
    pos[1] = excitation_cmd.angle1;
    pos[2] = angle;
    pos[3] = angle;
    pos[4] = excitation_cmd.angle4;
    pos[5] = excitation_cmd.angle5;
    pos[6] = excitation_cmd.angle6;
    chain_interface->move(pos);
    */

    static int seq_num;
    excitation_cmd.command = "";
    command_interface.getCommand(excitation_cmd, seq_num);

    if( excitation_cmd.command == "set_legs_refs") {
        //read right leg refs
        pos[0] = excitation_cmd.angle0;
        pos[1] = excitation_cmd.angle1;
        pos[2] = excitation_cmd.angle2;
        pos[3] = excitation_cmd.angle3;
        pos[4] = excitation_cmd.angle4;
        pos[5] = excitation_cmd.angle5;

        //read left leg refs
        pos[6] = excitation_cmd.angle6;
        pos[7] = excitation_cmd.angle7;
        pos[8] = excitation_cmd.angle8;
        pos[9] = excitation_cmd.angle9;
        pos[10] = excitation_cmd.angle10;
        pos[11] = excitation_cmd.angle11;

        //chain_interface->setReferenceSpeeds(vels);
        right_leg_chain_if->move(pos.subVector(0,5));
        left_leg_chain_if->move(pos.subVector(6,11));

        //chain_interface->move(vels);

        // get state data and send
        right_leg_chain_if->sensePosition(q_r);     //get positions in deg
        left_leg_chain_if->sensePosition(q_l);
        right_leg_chain_if->senseVelocity(qdot_r);  //get velocities in deg/s
        left_leg_chain_if->senseVelocity(qdot_l);
        right_leg_chain_if->senseTorque(tau_r);     //get torques in Nm
        left_leg_chain_if->senseTorque(tau_l);

        yarp::os::Bottle out_bottle;
        yarp::os::Bottle &out0 = out_bottle.addList();
        yarp::os::Bottle &out1 = out_bottle.addList();
        yarp::os::Bottle &out2 = out_bottle.addList();

        for(int i=0; i<q_r.size(); i++){
            out0.addDouble(q_r[i]);
            out0.addDouble(q_l[i]);
            out1.addDouble(qdot_r[i]);
            out1.addDouble(qdot_l[i]);
            out2.addDouble(tau_r[i]);
            out2.addDouble(tau_l[i]);
        }
        out_bottle.addDouble(t);

        if(outgoingPort.isOpen()) {
            outgoingPort.write(out_bottle);
        }
    }

    /*else if( excitation_cmd.command == "get_left_arm_measurements" ) {
        chain_interface->sensePosition(q);     //get positions in deg
        chain_interface->senseVelocity(qdot);  //get velocities in deg/s
        chain_interface->senseTorque(ta);     //get torques in Nm

        yarp::os::Bottle out_bottle;
        yarp::os::Bottle &out0 = out_bottle.addList();
        yarp::os::Bottle &out1 = out_bottle.addList();
        yarp::os::Bottle &out2 = out_bottle.addList();

        for(int i=0; i<q.size(); i++){
            out0.addDouble(q[i]);
            out1.addDouble(qdot[i]);
            out2.addDouble(tau[i]);
        }
        out_bottle.addDouble(t);

        if(outgoingPort.isOpen()) {
            outgoingPort.write(out_bottle);
        }

        yarp::sig::Vector out;
        out.resize(num_joints*3+1);
        out.setSubvector(0, q);
        out.setSubvector(num_joints, qdot);
        out.setSubvector(num_joints*2, tau);
        out[num_joints*3] = t;

        static VectorXd tmp(out.size());
        for (int i=0; i<tmp.rows(); i++)
          tmp(i) = out(i);
        tmpLog.push_back(tmp);
    }*/

    else if( excitation_cmd.command != "" ) {
        std::cout << excitation_cmd.command <<  " -> command not valid" << std::endl;
    }
}
