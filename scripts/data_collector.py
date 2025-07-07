#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import csv
import os

class DataCollector(Node):
    """
    ユーザーのキーボード操作（/cmd_vel）と、その時のLiDARデータ（/scan）をペアで記録し、
    CSVファイルに保存するデータ収集ノード。
    """
    def __init__(self):
        super().__init__('data_collector')
        
        self.file_path = os.path.expanduser('~/my_ros_project/data/training_data.csv')
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.csv_file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        self.latest_scan = None

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        self.get_logger().info('Data Collector node has been started. Ready to record driving data.')

    def scan_callback(self, msg):
        self.latest_scan = msg.ranges

    def cmd_vel_callback(self, msg):
        if self.latest_scan:
            row = [msg.linear.x, msg.angular.z] + list(self.latest_scan)
            self.writer.writerow(row)
    
    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info('Training data saved to: ' + self.file_path)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()