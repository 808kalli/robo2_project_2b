#!/usr/bin/env python
import rospy
import numpy as np
from math import sin, cos, atan2, pi, sqrt, asin
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time

from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

def quaternion_to_euler(w, x, y, z):
    """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def normalize_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        return self.x, self.P

class KalmanFilterNode:
    def __init__(self):
        rospy.init_node('kalman_filter_node')

        # Sensors
        self.imu = Imu()
        self.velocity = Twist()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        # Define the system parameters
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0.5], [1]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[0.01, 0], [0, 0.01]])
        self.R = np.array([[0.1]])
        x0 = np.array([[0], [1]])
        P0 = np.array([[1, 0], [0, 1]])

        # Initialize the Kalman filter
        self.kf = KalmanFilter(self.A, self.B, self.H, self.Q, self.R, x0, P0)

        self.u = np.array([[0]])  # Control input, assuming zero for simplicity


        #SUBSCRIBE
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1) 
        self.velocity_sub = rospy.Subscriber('/mymobibot/cmd_vel', Twist, self.vel_callback)

        #PUBLISH
        self.pub = rospy.Publisher('/filtered_pose', PoseStamped, queue_size=10)

    def vel_callback(self, msg):
        self.velocity = msg
        #self.velocity.angular.z for angular zvelocity
        #self.velocity.linear.x for linear x velocity
        # self.kf.predict(self.u)
        # x, P = self.kf.update(measurement)

        # # Create PoseStamped message
        # pose_msg = PoseStamped()
        # pose_msg.header.stamp = rospy.Time.now()
        # pose_msg.header.frame_id = "map"
        # pose_msg.pose.position.x = x[0, 0]
        # pose_msg.pose.position.y = 0
        # pose_msg.pose.position.z = 0
        # pose_msg.pose.orientation.x = 0
        # pose_msg.pose.orientation.y = 0
        # pose_msg.pose.orientation.z = 0
        # pose_msg.pose.orientation.w = 1

        # print("velocity: ", self.velocity.linear.x)

        # self.pub.publish(pose_msg)

    def imu_callback(self, msg):
        # ROS callback to get the /imu

        self.imu = msg
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

        # print("yaw: ", np.rad2deg(self.imu_yaw))

    def sonar_front_callback(self, msg):
        # ROS callback to get the /sensor/sonar_F

        self.sonar_F = msg
        print("front sonar: ", self.sonar_F.range)

if __name__ == '__main__':
    try:
        kf_node = KalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
