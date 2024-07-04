# robo2_project_2b
To launch:

- roslaunch mymobibot_gazebo mymobibot_world_loc.launch
- roslaunch robo2_mobile random_walk.launch

Comments for future changes:

- Sonar sensors of simulation only go up to 2m distance, so if d == 2.0 we should use other sensor for update step in EKF

