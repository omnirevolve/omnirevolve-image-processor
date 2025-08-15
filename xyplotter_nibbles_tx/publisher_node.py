import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
import time
import os


class ByteStreamPublisher(Node):
    def __init__(self):
        super().__init__('byte_stream_publisher')

        self.declare_parameter('input_file', '')
        self.declare_parameter('packet_size', 1024)
        self.declare_parameter('send_delay_ms', 5)
        self.declare_parameter('use_seq_id', True)

        self.input_file = self.get_parameter('input_file').get_parameter_value().string_value
        self.packet_size = self.get_parameter('packet_size').get_parameter_value().integer_value
        self.send_delay = self.get_parameter('send_delay_ms').get_parameter_value().integer_value / 1000.0
        self.use_seq_id = self.get_parameter('use_seq_id').get_parameter_value().bool_value

        if not self.input_file or not os.path.isfile(self.input_file):
            self.get_logger().error(f"Invalid input_file: {self.input_file}")
            rclpy.shutdown()
            return

        if self.use_seq_id and self.packet_size < 2:
            self.get_logger().error("packet_size must be >= 2 when use_seq_id is True")
            rclpy.shutdown()
            return

        self.publisher_ = self.create_publisher(
            UInt8MultiArray,
            '/plotter/byte_stream',
            rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        )

        self.get_logger().info(f"Starting stream from {self.input_file}, packet_size={self.packet_size}, delay={self.send_delay}s, seq_id={self.use_seq_id}")
        self.publish_stream()

    def publish_stream(self):
        seq_id = 0
        with open(self.input_file, 'rb') as f:
            while True:
                payload_size = self.packet_size - (1 if self.use_seq_id else 0)
                chunk = f.read(payload_size)
                if not chunk:
                    self.get_logger().info("End of file reached.")
                    break

                if self.use_seq_id:
                    data = bytes([seq_id]) + chunk
                    seq_id = (seq_id + 1) & 0xFF
                else:
                    data = chunk

                msg = UInt8MultiArray()
                msg.data = list(data)
                self.publisher_.publish(msg)

                time.sleep(self.send_delay)


def main(args=None):
    rclpy.init(args=args)
    node = ByteStreamPublisher()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
