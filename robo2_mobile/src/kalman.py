#!/usr/bin/env python
import rospy
import numpy as np
from math import sin, cos, atan2, pi, sqrt, asin, acos
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

class KalmanFilterNode():
    def __init__(self, rate):
        rospy.init_node('kalman_filter_node')
        self.pub_rate = rospy.Rate(rate)

        # Sensors
        self.imu = Imu()
        self.velocity = Twist()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        #System Parameters
        self.x = np.array([[0.0], [0.0], [0.0]])
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0.5], [1]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[0.01, 0], [0, 0.01]])
        self.R = np.array([[0.1]])
        x0 = np.array([[0], [1]])
        P0 = np.array([[1, 0], [0, 1]])

        self.u = np.array([[0]])  # Control input, assuming zero for simplicity


        #SUBSCRIBE
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)
        self.velocity_sub = rospy.Subscriber('/mymobibot/cmd_vel', Twist, self.vel_callback)

        #PUBLISH
        self.pub = rospy.Publisher('/filtered_pose', PoseStamped, queue_size=10)

        self.publish()

    def vel_callback(self, msg):
        self.velocity = msg
        # self.velocity.angular.z for angular zvelocity
        # self.velocity.linear.x for linear x velocity
        # print("velocity: ", self.velocity.linear.x)

    def imu_callback(self, msg):
        self.imu = msg
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
        # print("yaw: ", np.rad2deg(self.imu_yaw))

    def sonar_front_callback(self, msg):
        self.sonar_F = msg

    def sonar_frontleft_callback(self, msg):
        self.sonar_FL = msg

    def sonar_frontright_callback(self, msg):
        self.sonar_FR = msg

    def sonar_left_callback(self, msg):
        self.sonar_L = msg

    def sonar_right_callback(self, msg):
        self.sonar_R = msg

    # def predict(self):
    #     self.x = self.A @ self.x + self.B @ u
    #     self.P = self.A @ self.P @ self.A.T + self.Q
    #     return self.x, self.P

    # def update(self):
    #     y = z - self.H @ self.x
    #     S = self.H @ self.P @ self.H.T + self.R
    #     K = self.P @ self.H.T @ np.linalg.inv(S)
    #     self.x = self.x + K @ y
    #     self.P = self.P - K @ self.H @ self.P
    #     return self.x, self.P

    def publish(self):

        fsfw = 0.35
        flsfw = 0.25
        frsfw = 0.25
        phase = 1

        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()

        print("Executing Extended Kalman Filter...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()

        while not rospy.is_shutdown():

            sonar_front = self.sonar_F.range
            sonar_front_left = self.sonar_FL.range
            sonar_front_right = self.sonar_FR.range
            sonar_left = self.sonar_L.range
            sonar_right = self.sonar_R.range

            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9

            # print(dt)

            self.x = self.x + np.array([[self.velocity.linear.x*dt*cos(self.x[2])], [self.velocity.linear.x*dt*sin(self.x[2])], [-self.velocity.angular.z*dt]])
            self.x[2] = (self.x[2] + pi) % (2*pi) - pi      #normalize angle
            print(self.x)

            if (phase == 1):
                if ((sonar_front < fsfw) or (sonar_front_left < flsfw) or (sonar_front_right < frsfw)):
                    phase = 2
            if (phase == 2):
                if (sonar_front < 2.0) and ((sonar_left < 2.0) or (sonar_right < 2.0)):
                    d1 = sonar_front*abs(cos(self.imu_yaw))
                    d2 = sonar_left*abs(cos(self.imu_yaw)) if sonar_left < sonar_right else sonar_right*abs(cos(self.imu_yaw))
                else:
                    d1 = None
                    d2 = None

                if ((d1 is not None) and (d2 is not None)):
                    if ((self.x[0] > 0) and (self.x[1] > 0)):       #Q2  
                        self.z = np.array([[2-d1], [2-d2], [self.imu_yaw]])
                    if ((self.x[0] < 0) and (self.x[1] > 0)):       #Q3  
                        self.z = np.array([[-2+d1], [2-d2], [self.imu_yaw]])
                    if ((self.x[0] > 0) and (self.x[1] < 0)):       #Q1  
                        self.z = np.array([[2-d1], [-2+d2], [self.imu_yaw]])
                    if ((self.x[0] < 0) and (self.x[1] < 0)):       #Q4  
                        self.z = np.array([[-2+d1], [-2+d2], [self.imu_yaw]])            


            # #EKF 
            # self.kf.predict(self.u)
            # x, P = self.kf.update(measurement)

            # # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = self.x[0]
            pose_msg.pose.position.y = self.x[1]
            # pose_msg.pose.position.z = 0
            # pose_msg.pose.orientation.x = 0
            # pose_msg.pose.orientation.y = 0
            # pose_msg.pose.orientation.z = 0
            # pose_msg.pose.orientation.w = 1

            self.pub.publish(pose_msg)

            self.pub_rate.sleep()


if __name__ == '__main__':
    try:
        rate = rospy.get_param("/kalman/rate")
        kf_node = KalmanFilterNode(rate)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
