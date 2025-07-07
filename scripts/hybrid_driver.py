#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import numpy as np
import joblib
import os
from tf_transformations import euler_from_quaternion

class HybridDriver(Node):
    """
    2つのAIを統合し、状況に応じて判断を切り替える司令塔（アービター）ノード。
    """
    def __init__(self):
        super().__init__('hybrid_driver')
        
        # 学習済み模倣学習モデルをロード
        model_path = os.path.expanduser('~/my_ros_project/models/imitation_model.pkl')
        self.imitation_model = joblib.load(model_path)
        self.get_logger().info(f'Loaded imitation learning model from {model_path}')

        # 変数の初期化
        self.current_pose = None
        self.current_yaw = 0.0
        self.goal_x = 5.0
        self.goal_y = 0.0
        self.safe_distance = 1.0 # 回避行動を開始する安全距離(m)

        # ROS 2通信の設定
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('Hybrid Driver node has been started.')

    def odom_callback(self, msg):
        # ロボットの自己位置を更新
        self.current_pose = msg.pose.pose
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.current_yaw) = euler_from_quaternion(orientation_list)

    def scan_callback(self, msg):
        if self.current_pose is None:
            return

        twist = Twist()
        
        # 正面30度の範囲で、最も近い障害物との距離を計算
        front_ranges = msg.ranges[345:] + msg.ranges[:16]
        closest_front_distance = min(front_ranges)
        
        # --- 司令塔(Arbiter)の判断ロジック ---
        if closest_front_distance > self.safe_distance:
            # 【モード①：ゴール追跡】正面が安全な場合
            
            # ゴールへの方角と距離を計算
            dx = self.goal_x - self.current_pose.position.x
            dy = self.goal_y - self.current_pose.position.y
            angle_to_goal = math.atan2(dy, dx)
            angle_diff = angle_to_goal - self.current_yaw
            
            # 角度差を正規化
            while angle_diff > math.pi: angle_diff -= 2 * math.pi
            while angle_diff < -math.pi: angle_diff += 2 * math.pi

            # ゴールを向いていなければ回転、向いていれば前進
            if abs(angle_diff) > 0.2:
                twist.angular.z = 0.3
            else:
                twist.linear.x = 0.1
            
            self.get_logger().info(f'Mode: Goal Tracking | Closest Front Obstacle: {closest_front_distance:.2f}')

        else:
            # 【モード②：障害物回避】正面が危険な場合
            
            # 模倣学習AIに判断を委ねる
            scan_data = np.array(msg.ranges).reshape(1, -1)
            scan_data[np.isinf(scan_data)] = 3.5
            scan_data[np.isnan(scan_data)] = 0
            predicted_velocities = self.imitation_model.predict(scan_data)
            
            twist.linear.x = predicted_velocities[0, 0]
            twist.angular.z = predicted_velocities[0, 1]
            
            self.get_logger().info(f'Mode: Obstacle Avoidance (AI) | Closest Front Obstacle: {closest_front_distance:.2f}')
        
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = HybridDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 終了時にロボットを停止させる
        stop_twist = Twist()
        node.publisher_.publish(stop_twist)
        node.get_logger().info('Robot stopped.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()