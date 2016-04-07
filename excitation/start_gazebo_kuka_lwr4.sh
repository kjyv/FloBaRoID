#!/bin/sh

#roslaunch single_lwr_launch single_lwr.launch load_moveit:=true
roslaunch single_lwr_launch single_lwr.launch load_moveit:=true use_lwr_sim:=false lwr_powered:=true
