import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(depth=10)
qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

class WebcamListener(Node):
    def __init__(self):
        super().__init__('webcam_listener')
        print("Hello there!")
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, qos_profile)
        self.bridge = CvBridge()

    def callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv2.imshow("Live Feed", img)
            if cv2.waitKey(1) == 27:
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    node = WebcamListener()
    rclpy.spin(node)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
