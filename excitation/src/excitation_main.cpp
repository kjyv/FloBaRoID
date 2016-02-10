#include <yarp/os/all.h>
#include <GYM/generic_module.hpp>
#include "excitation_module.hpp"

// default module period
#define MODULE_PERIOD 5 //[millisec]

int main(int argc, char* argv[])
{
    // yarp network declaration and check
    yarp::os::Network yarp;
    if(!yarp.checkNetwork()){
        std::cerr <<"yarpserver not running - run yarpserver"<< std::endl;
        exit(EXIT_FAILURE);
    }
    // yarp network initialization
    yarp.init();

    // create rf
    yarp::os::ResourceFinder rf;
    rf.setVerbose(true);
    // set excitation_initial_config.ini as default
    // to specify another config file, run with this arg: --from your_config_file.ini
    rf.setDefaultConfigFile( "excitation_initial_config.ini" );
    rf.setDefaultContext( "excitation" );
    rf.configure(argc, argv);
    // create my module
    excitation_module excitation_mod = excitation_module( argc, argv, "excitation", MODULE_PERIOD, rf );

    // run the module
    excitation_mod.runModule( rf );

    // yarp network deinitialization
    yarp.fini();

    exit(EXIT_SUCCESS);
}
