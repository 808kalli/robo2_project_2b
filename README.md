# robo2_project_2b
To launch:

- roslaunch mymobibot_gazebo mymobibot_world_loc.launch
- roslaunch robo2_mobile random_walk.launch

To record to bag files:
- For filter output:
    - rosbag record /filtered_pose
    - rtopic echo -b bag_file_name.bag -p /filtered_pose > data_pred.txt

- For real output:
    - rosbag record /mymobibot/odom
    - rtopic echo -b bag_file_name.bag -p /mymobibot/odom > data_real.txt

Comments for future changes:

- Sonar sensors of simulation only go up to 2m distance, so if d == 2.0 we should use other sensor for update step in EKF

