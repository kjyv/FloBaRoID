import sys
import numpy as np
import threading

import rospy
import moveit_commander
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

from trajectoryGenerator import TrajectoryGenerator

# records the states obtained from joint_states messages
class RecordJointStates(object):
    def __init__(self):
        #rospy.init_node('joint_states_listener')
        #self.lock = threading.Lock()
        self.name = []
        self.positions = []
        self.velocities = []
        self.torques = []
        self.times = []
        self.next = True
        self.listen()

    def listen(self):
        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)

    #callback function: when a joint_states message arrives, save the values
    def joint_states_callback(self, msg):
        self.name = msg.name
        self.positions.append(msg.position)
        self.velocities.append(msg.velocity)
        self.torques.append(msg.effort)   #ros "effort" is force for linear or torque for rotational joints
        self.times.append(msg.header.stamp.secs + msg.header.stamp.nsecs / 1.0e9)

def main(config, data):
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('excitation_move_group', anonymous=False)
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("full_lwr")

    # in case there are previous executions still running
    group.stop()

    # reset to homing position
    group.set_named_target('full_lwr_home')
    group.go()

    group.allow_replanning(True)

    # set higher goal tolerance in case moveit aborts with GOAL_TOLERANCE_VIOLATED.
    # driver has its own tol as well?
    #group.set_goal_tolerance(0.1)
    #group.set_goal_joint_tolerance(0.1)
    #group.set_goal_position_tolerance(0.1)

    config['N_DOFS'] = len(group.get_current_joint_values())
    trajectories = TrajectoryGenerator(config['N_DOFS'])

    plan = group.plan()
    plan.joint_trajectory.points = []

    # generate trajectory and send in one message
    duration = config['args'].periods*trajectories.getPeriodLength()
    sent_positions = list()
    sent_time = list()
    sent_velocities = list()
    sent_accelerations = list()

    t = 0
    step = 1.0/200   # data rate of 200 Hz
    while t < duration:
        trajectories.setTime(t)
        point = JointTrajectoryPoint()
        for i in range(0, config['N_DOFS']):
            q = trajectories.getAngle(i)
            point.positions.append(q)
            dq = trajectories.getVelocity(i)
            point.velocities.append(dq)
            ddq = trajectories.getAcceleration(i)
            point.accelerations.append(ddq)

        point.time_from_start = rospy.Duration(t)
        plan.joint_trajectory.points.append(point)

        sent_positions.append(point.positions)
        sent_velocities.append(point.velocities)
        sent_accelerations.append(point.accelerations)
        sent_time.append(t)
        t+=step

    jSt = RecordJointStates()
    group.execute(plan, wait=False)
    num_sent = len(sent_positions)
    start_t = rospy.get_time()
    while len(jSt.positions) < num_sent: # and rospy.get_time() < start_t+duration
        # gets data in thread
        rospy.sleep(step)

    data['Q'] = np.array(jSt.positions[0:num_sent])
    data['V'] = np.array(jSt.velocities[0:num_sent])
    data['T'] = np.array(jSt.times[0:num_sent])
    data['Tau'] = np.array(jSt.torques[0:num_sent])
    data['measured_frequency'] = data['Q'].shape[0] / duration
    data['Qsent'] = np.array(sent_positions);
    data['QdotSent'] = np.array(sent_velocities);
    data['QddotSent'] = np.array(sent_accelerations);

    print "got {} samples in {}s.".format(data['Q'].shape[0], duration),
    print "(about {} Hz)".format(data['measured_frequency'])

if __name__ == '__main__':
    main({}, {})
