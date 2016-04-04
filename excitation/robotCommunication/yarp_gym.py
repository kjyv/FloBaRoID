import sys
import yarp
import numpy as np
from trajectoryGenerator import TrajectoryGenerator

def gen_position_msg(msg_port, angles, velocities):
    bottle = msg_port.prepare()
    bottle.clear()
    bottle.fromString("(set_left_arm {} {}) 0".format(' '.join(map(str, angles)), ' '.join(map(str, velocities)) ))
    return bottle

def gen_command(msg_port, command):
    bottle = msg_port.prepare()
    bottle.clear()
    bottle.fromString("({}) 0".format(command))
    return bottle

def main(config, data):
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

    # init trajectory generator for all the joints
    trajectories = TrajectoryGenerator(config['N_DOFS'], use_deg=True)

    t_init = yarp.Time.now()
    t_elapsed = 0.0
    duration = config['args'].periods*trajectories.getPeriodLength()   #init overall run duration to a periodic length

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
    #e = [0] * config['N_DOFS']
    #velocity_correction = [0] * config['N_DOFS']

    def wait_for_zero_vel(t_elapsed, trajectories):
        trajectories.setTime(t_elapsed)
        if abs(round(trajectories.getVelocity(0))) < 5:
            return True

    waited_for_start = 0
    started = False
    while t_elapsed < duration:
        trajectories.setTime(t_elapsed)
        target_angles = [trajectories.getAngle(i) for i in range(0, config['N_DOFS'])]
        target_velocities = [trajectories.getVelocity(i) for i in range(0, config['N_DOFS'])]
        target_accelerations = [trajectories.getAcceleration(i) for i in range(0, config['N_DOFS'])]
        #for i in range(0, config['N_DOFS']):
        #    target_velocities[i]+=velocity_correction[i]

        # make sure we start moving at a position with zero velocity
        if not started:
            started = wait_for_zero_vel(t_elapsed, trajectories)
            t_elapsed = yarp.Time.now() - t_init
            waited_for_start = t_elapsed

            if started:
                # set angles and wait one period to have settled at zero velocity position
                gen_position_msg(command_port, target_angles, target_velocities)
                command_port.write()

                print "waiting to arrive at an initial position...",
                sys.stdout.flush()
                yarp.Time.delay(trajectories.getPeriodLength())
                t_init+=trajectories.getPeriodLength()
                duration+=waited_for_start
                print "ok."
            continue

        # set target angles
        gen_position_msg(command_port, target_angles, target_velocities)
        command_port.write()

        sent_positions.append(target_angles)
        sent_velocities.append(target_velocities)
        sent_accelerations.append(target_accelerations)
        sent_time.append(yarp.Time.now())

        # get and wait for next value, so sync to GYM loop
        data = data_port.read(shouldWait=True)

        b_positions = data.get(0).asList()
        b_velocities = data.get(1).asList()
        b_torques = data.get(2).asList()
        d_time = data.get(3).asDouble()

        positions = np.zeros(config['N_DOFS'])
        velocities = np.zeros(config['N_DOFS'])
        accelerations = np.zeros(config['N_DOFS'])
        torques = np.zeros(config['N_DOFS'])

        if config['N_DOFS'] == b_positions.size():
            for i in range(0, config['N_DOFS']):
                positions[i] = b_positions.get(i).asDouble()
                velocities[i] = b_velocities.get(i).asDouble()
                torques[i] = b_torques.get(i).asDouble()
        else:
            print "warning, wrong amount of values received! ({} DOFS vs. {})".format(config['N_DOFS'], b_positions.size())

        # test manual correction for position error
        #p = 0
        #for i in range(0, config['N_DOFS']):
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
    data['Q'] = np.array(measured_positions); del measured_positions
    data['Qsent'] = np.array(sent_positions);
    data['QdotSent'] = np.array(sent_velocities);
    data['QddotSent'] = np.array(sent_accelerations);
    data['V'] = np.array(measured_velocities); del measured_velocities
    data['Tau'] = np.array(measured_torques); del measured_torques
    data['T'] = np.array(measured_time); del measured_time

    data['measured_frequency'] = len(sent_positions)/duration

    # some stats
    print "got {} samples in {}s.".format(data['Q'].shape[0], duration),
    print "(about {} Hz)".format(data['measured_frequency'])
