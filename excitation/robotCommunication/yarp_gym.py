from __future__ import division
from __future__ import print_function
from builtins import map
from builtins import range
import sys
import yarp
import numpy as np

def gen_position_msg(msg_port, angles):
    bottle = msg_port.prepare()
    bottle.clear()
    angles_right, angles_left = angles[0:6], angles[6:]
    bottle.fromString("(set_legs_refs {} {}) 0".format(' '.join(map(str, angles_right)), ' '.join(map(str, angles_left)) ))
    return bottle

def gen_command(msg_port, command):
    bottle = msg_port.prepare()
    bottle.clear()
    bottle.fromString("({}) 0".format(command))
    return bottle

def main(config, trajectory, out):
    # connect to yarp and open output port
    yarp.Network.init()
    yarp.Time.useNetworkClock("/clock")
    yarp.Time.now()  #use clock once to sync (?)
    while not yarp.Time.isValid():
        continue

    portName = '/excitation/command:'
    command_port = yarp.BufferedPortBottle()
    command_port.open(portName+'o')
    yarp.Network.connect(portName+'o', portName+'i')

    portName = '/excitation/state:'
    data_port = yarp.BufferedPortBottle()
    data_port.open(portName+"i")
    yarp.Network.connect(portName+'o', portName+'i')

    t_init = yarp.Time.now()
    t_elapsed = 0.0
    duration = config['args'].periods*trajectory.getPeriodLength()   #init overall run duration to a periodic length

    measured_positions = list()
    measured_velocities = list()
    measured_accelerations = list()
    measured_torques = list()
    measured_time = list()

    sent_positions = list()
    sent_time = list()
    sent_velocities = list()
    sent_accelerations = list()

    # try high level p correction when using velocity ctrl
    #e = [0] * config['num_dofs']
    #velocity_correction = [0] * config['num_dofs']

    waited_for_start = 0
    started = False
    while t_elapsed < duration:
        trajectory.setTime(t_elapsed)
        target_angles = [trajectory.getAngle(i) for i in range(0, config['num_dofs'])]
        target_velocities = [trajectory.getVelocity(i) for i in range(0, config['num_dofs'])]
        target_accelerations = [trajectory.getAcceleration(i) for i in range(0, config['num_dofs'])]
        #for i in range(0, config['num_dofs']):
        #    target_velocities[i]+=velocity_correction[i]

        # make sure we start moving at a position with zero velocity
        if not started:
            started = trajectory.wait_for_zero_vel(t_elapsed)
            t_elapsed = yarp.Time.now() - t_init
            waited_for_start = t_elapsed

            if started:
                # set angles and wait one period to have settled at zero velocity position
                gen_position_msg(command_port, target_angles)
                command_port.write()

                print("waiting to arrive at an initial position...", end=' ')
                sys.stdout.flush()
                yarp.Time.delay(trajectory.getPeriodLength())
                t_init+=trajectory.getPeriodLength()
                duration+=waited_for_start
                print("ok.")
            continue

        # set target angles
        gen_position_msg(command_port, target_angles)
        command_port.write()

        sent_positions.append(target_angles)
        sent_velocities.append(target_velocities)
        sent_accelerations.append(target_accelerations)
        sent_time.append(yarp.Time.now())

        # get and wait for next value, so sync to GYM loop
        data_out = data_port.read(shouldWait=True)

        b_positions = data_out.get(0).asList()
        b_velocities = data_out.get(1).asList()
        b_torques = data_out.get(2).asList()
        d_time = data_out.get(3).asDouble()

        positions = np.zeros(config['num_dofs'])
        velocities = np.zeros(config['num_dofs'])
        accelerations = np.zeros(config['num_dofs'])
        torques = np.zeros(config['num_dofs'])

        if config['num_dofs'] == b_positions.size():
            for i in range(0, config['num_dofs']):
                positions[i] = b_positions.get(i).asDouble()
                velocities[i] = b_velocities.get(i).asDouble()
                torques[i] = b_torques.get(i).asDouble()
        else:
            print("warning, wrong amount of values received! ({} DOFS vs. {})".format(config['num_dofs'], b_positions.size()))

        # test manual correction for position error
        #p = 0
        #for i in range(0, config['num_dofs']):
        #    e[i] = (angles[i] - positions[i])
        #    velocity_correction[i] = e[i]*p

        # collect measurement data
        measured_positions.append(positions)
        measured_velocities.append(velocities)
        measured_torques.append(torques)
        measured_time.append(d_time)
        t_elapsed = d_time - t_init

    # clean up
    command_port.close()
    data_port.close()
    out['Q'] = np.array(measured_positions); del measured_positions
    out['Qsent'] = np.array(sent_positions);
    out['QdotSent'] = np.array(sent_velocities);
    out['QddotSent'] = np.array(sent_accelerations);
    out['V'] = np.array(measured_velocities); del measured_velocities
    out['Tau'] = np.array(measured_torques); del measured_torques
    out['T'] = np.array(measured_time); del measured_time

    out['measured_frequency'] = len(sent_positions)/duration

    # some stats
    print("got {} samples in {}s.".format(out['Q'].shape[0], duration), end=' ')
    print("(about {} Hz)".format(out['measured_frequency']))
