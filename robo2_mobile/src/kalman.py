#!/usr/bin/env python
import rospy
import numpy as np
from math import sin, cos, atan2, pi, sqrt, asin, acos
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time
from numpy.linalg import inv

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

class KalmanFilterNode():
    def __init__(self, rate):
        rospy.init_node('kalman_filter_node')
        self.pub_rate = rospy.Rate(rate)

        # Sensors
        self.imu = Imu()
        self.velocity = Twist()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.imuAngVelZ = 0.0
        self.imuLinAccX = 0.0
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        #System Parameters
        self.x = np.array([[0.0], [0.0], [2.57]])
        self.vel = 0.25

        #Which wall each sonar is pointing to, 0 means no wall detected (distance >= 2.0)
        self.wall_f = 0
        self.wall_fl = 0
        self.wall_l = 0
        self.wall_fr = 0
        self.wall_r = 0

        self.H = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        
        self.R = np.array([[0.001, 0.0, 0.0],  #sensor covarriance matrix
                           [0.0, 0.001, 0.0],
                           [0.0, 0.0, 0.00004]])

        self.P = np.array([[0.00001, 0, 0],     #starting P array, we are very ceratin about the starting position
                           [0, 0.00001, 0],
                           [0, 0, 0.00001]])

        self.z = np.array([[0.0], [0.0], [0.0]])


        #SUBSCRIBE
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)
        self.velocity_sub = rospy.Subscriber('/mymobibot/cmd_vel', Twist, self.vel_callback)

        #PUBLISH
        self.pose_pub = rospy.Publisher('/filtered_pose', PoseStamped, queue_size=10)
        self.yaw_pub = rospy.Publisher('/filtered_yaw', Float64, queue_size=10)

        self.publish()

    def vel_callback(self, msg):
        self.velocity = msg
        # self.velocity.angular.z for angular zvelocity
        # self.velocity.linear.x for linear x velocity
        # print("velocity: ", self.velocity.linear.x)

    def imu_callback(self, msg):
        self.imu = msg
        self.imuAngVelZ = msg.angular_velocity.z
        self.imuLinAccX = msg.linear_acceleration.x
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

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

    def predict(self, dt):

        self.x = self.x + np.array([[self.vel*dt*cos(self.x[2]) + 0.5*self.imuLinAccX*(dt**2)*cos(self.x[2])], [self.vel*dt*sin(self.x[2]) + 0.5*self.imuLinAccX*(dt**2)*sin(self.x[2])], [self.imuAngVelZ*dt]])
        self.vel = self.vel + self.imuLinAccX*dt
        if self.vel < 0.0: 
            self.vel = 0.0
        if self.vel > 0.25:
            self.vel = 0.25
        self.x[2] = (self.x[2] + pi) % (2*pi) - pi      #normalize angle

        self.A = np.array([[1, 0, -self.vel*dt*sin(self.x[2]) - 0.5*self.imuLinAccX*(dt**2)*sin(self.x[2])],
                           [0, 1, self.vel*dt*cos(self.x[2]) + 0.5*self.imuLinAccX*(dt**2)*cos(self.x[2])],
                           [0, 0, 1]])
        
        self.Cw = np.array([[0.00001, 0, 0],
                           [0, 0.00001, 0],
                           [0, 0, 0.2]])
                
        self.P = self.A@self.P@self.A.T + self.Cw

        return self.x, self.P

    def update(self, sonar_front, sonar_front_left, sonar_front_right, sonar_left, sonar_right):

        dx = cos(self.x[2]) * 0.2
        dy = sin(self.x[2]) * 0.2

        angle_fl = (self.x[2] + pi/4 + pi) % (2*pi) - pi
        angle_l = (self.x[2] + pi/2 + pi) % (2*pi) - pi
        angle_fr = (self.x[2] - pi/4 + pi) % (2*pi) - pi
        angle_r = (self.x[2] - pi/2 + pi) % (2*pi) - pi
        
        top_left = pi/2 - atan2((2 - dx - self.x[0]), (2 - dy - self.x[1]))
        bottom_left = pi/2 - atan2((-2 - dx - self.x[0]), (2 - dy - self.x[1]))
        top_right = 2*pi + atan2((-2 - dy - self.x[1]), (2 - dx - self.x[0]))
        bottom_right = 2*pi + atan2((-2 - dy - self.x[1]), (-2 - dx - self.x[0]))

        top_left = (top_left + pi) % (2*pi) - pi                #normalize angle
        bottom_left = (bottom_left + pi) % (2*pi) - pi          #normalize angle
        top_right = (top_right + pi) % (2*pi) - pi              #normalize angle
        bottom_right = (bottom_right + pi) % (2*pi) - pi        #normalize angle

        #front sonar wall
        if (self.x[2] <= top_left and self.x[2] > top_right):                                                           #wall 1
            self.wall_f = 1
        elif (self.x[2] <= bottom_left and self.x[2] > top_left):                                                       #wall 2
            self.wall_f = 2
        elif ((self.x[2] <= bottom_right and self.x[2] > -pi) or (self.x[2] > bottom_left and self.x[2] <= pi)):        #wall 3
            self.wall_f = 3
        elif (self.x[2] <= top_right and self.x[2] > bottom_right):                                                     #wall 4
            self.wall_f = 4

        #front_left sonar wall
        if (angle_fl <= top_left and angle_fl > top_right):                                                             #wall 1
            self.wall_fl = 1
        elif (angle_fl <= bottom_left and angle_fl > top_left):                                                         #wall 2
            self.wall_fl = 2
        elif ((angle_fl <= bottom_right and angle_fl > -pi) or (angle_fl > bottom_left and angle_fl <= pi)):            #wall 3
            self.wall_fl = 3
        elif (angle_fl <= top_right and angle_fl > bottom_right):                                                       #wall 4
            self.wall_fl = 4

        #left sonar wall
        if (angle_l <= top_left and angle_l > top_right):                                                               #wall 1
            self.wall_l = 1
        elif (angle_l <= bottom_left and angle_l > top_left):                                                           #wall 2
            self.wall_l = 2
        elif ((angle_l <= bottom_right and angle_l > -pi) or (angle_l > bottom_left and angle_l <= pi)):                #wall 3
            self.wall_l = 3
        elif (angle_l <= top_right and angle_l > bottom_right):                                                         #wall 4
            self.wall_l = 4

        #front_right sonar wall
        if (angle_fr <= top_left and angle_fr > top_right):                                                             #wall 1
            self.wall_fr = 1
        elif (angle_fr <= bottom_left and angle_fr > top_left):                                                         #wall 2
            self.wall_fr = 2
        elif ((angle_fr <= bottom_right and angle_fr > -pi) or (angle_fr > bottom_left and angle_fr <= pi)):            #wall 3
            self.wall_fr = 3
        elif (angle_fr <= top_right and angle_fr > bottom_right):                                                       #wall 4
            self.wall_fr = 4

        #right sonar wall
        if (angle_r <= top_left and angle_r > top_right):                                                               #wall 1
            self.wall_r = 1
        elif (angle_r <= bottom_left and angle_r > top_left):                                                           #wall 2
            self.wall_r = 2
        elif ((angle_r <= bottom_right and angle_r > -pi) or (angle_r > bottom_left and angle_r <= pi)):                #wall 3
            self.wall_r = 3
        elif (angle_r <= top_right and angle_r > bottom_right):                                                         #wall 4
            self.wall_r = 4

        
        #find position based on measurements
        #wall 1
        if (sonar_front < 1.9 and self.wall_f == 1):
            d1 = (sonar_front + 0.2)*abs(cos(self.x[2]))
            self.z[0][0] = 2 - d1
        elif (sonar_front_left < 1.9 and self.wall_fl == 1):
            d1 = (sonar_front_left + 0.14142135623)*abs(cos(angle_fl))
            self.z[0][0] = 2 - d1
        elif (sonar_front_right < 1.9 and self.wall_fr == 1):
            d1 = (sonar_front_right + 0.14142135623)*abs(cos(angle_fr))
            self.z[0][0] = 2 - d1
        elif (sonar_left < 1.9 and self.wall_l == 1):
            d1 = (sonar_left + 0.1)*abs(cos(angle_l))
            self.z[0][0] = 2 - d1
        elif (sonar_right < 1.9 and self.wall_r == 1):
            d1 = (sonar_right + 0.1)*abs(cos(angle_r))
            self.z[0][0] = 2 - d1
        #wall 3
        elif (sonar_front < 1.9 and self.wall_f == 3):
            d1 = (sonar_front + 0.2)*abs(cos(self.x[2]))
            self.z[0][0] = -2 + d1
        elif (sonar_front_left < 1.9 and self.wall_fl == 3):
            d1 = (sonar_front_left + 0.14142135623)*abs(cos(angle_fl))
            self.z[0][0] = -2 + d1
        elif (sonar_front_right < 1.9 and self.wall_fr == 3):
            d1 = (sonar_front_right + 0.14142135623)*abs(cos(angle_fr))
            self.z[0][0] = -2 + d1
        elif (sonar_left < 1.9 and self.wall_l == 3):
            d1 = (sonar_left + 0.1)*abs(cos(angle_l))
            self.z[0][0] = -2 + d1
        elif (sonar_right < 1.9 and self.wall_r == 3):
            d1 = (sonar_right + 0.1)*abs(cos(angle_r))
            self.z[0][0] = -2 + d1
        else:
            d1 = None

        #wall 2
        if (sonar_front < 1.9 and self.wall_f == 2):
            d2 = (sonar_front + 0.2)*abs(sin(self.x[2]))
            self.z[1][0] = 2 - d2
        elif (sonar_front_left < 1.9 and self.wall_fl == 2):
            d2 = (sonar_front_left + 0.14142135623)*abs(sin(angle_fl))
            self.z[1][0] = 2 - d2
        elif (sonar_front_right < 1.9 and self.wall_fr == 2):
            d2 = (sonar_front_right + 0.14142135623)*abs(sin(angle_fr))
            self.z[1][0] = 2 - d2
        elif (sonar_left < 1.9 and self.wall_l == 2):
            d2 = (sonar_left + 0.1)*abs(sin(angle_l))
            self.z[1][0] = 2 - d2
        elif (sonar_right < 1.9 and self.wall_r == 2):
            d2 = (sonar_right + 0.1)*abs(sin(angle_r))
            self.z[1][0] = 2 - d2
        #wall 4
        elif (sonar_front < 1.9 and self.wall_f == 4):
            d2 = (sonar_front + 0.2)*abs(sin(self.x[2]))
            self.z[1][0] = -2 + d2
        elif (sonar_front_left < 1.9 and self.wall_fl == 4):
            d2 = (sonar_front_left + 0.14142135623)*abs(sin(angle_fl))
            self.z[1][0] = -2 + d2
        elif (sonar_front_right < 1.9 and self.wall_fr == 4):
            d2 = (sonar_front_right + 0.14142135623)*abs(sin(angle_fr))
            self.z[1][0] = -2 + d2
        elif (sonar_left < 1.9 and self.wall_l == 4):
            d2 = (sonar_left + 0.1)*abs(sin(angle_l))
            self.z[1][0] = -2 + d2
        elif (sonar_right < 1.9 and self.wall_r == 4):
            d2 = (sonar_right + 0.1)*abs(sin(angle_r))
            self.z[1][0] = -2 + d2
        else:
            d2 = None

        self.z[2][0] = self.imu_yaw

        #measurement model if i only have measurement for x
        if ((d1 is not None) and (d2 is None)):
            self.z[1][0] = self.x[1][0]
            y = self.z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P

        #measurement model if i only have measurement for y
        if ((d1 is None) and (d2 is not None)):
            self.z[0][0] = self.x[0][0]
            y = self.z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P

        #measurement model if i have both x and y measurements
        if ((d1 is not None) and (d2 is not None)):
            y = self.z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P
        
        return self.x, self.P

    def publish(self):

        dt = 0.1

        print("Executing Extended Kalman Filter...")

        while not rospy.is_shutdown():

            sonar_front = self.sonar_F.range
            sonar_front_left = self.sonar_FL.range
            sonar_front_right = self.sonar_FR.range
            sonar_left = self.sonar_L.range
            sonar_right = self.sonar_R.range

            (self.x, self.P) = self.predict(dt)
            (self.x, self.P) = self.update(sonar_front, sonar_front_left, sonar_front_right, sonar_left, sonar_right)     

            #create coordinates message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = self.x[0]
            pose_msg.pose.position.y = self.x[1]

            #create yaw message
            yaw_msg = Float64()
            yaw_msg.data = self.x[2]

            #publish
            self.pose_pub.publish(pose_msg)
            self.yaw_pub.publish(yaw_msg)

            self.pub_rate.sleep()


if __name__ == '__main__':
    try:
        rate = rospy.get_param("/kalman/rate")
        kf_node = KalmanFilterNode(rate)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
