#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

class WebcamListener : public rclcpp::Node
{
public:
    WebcamListener() :
        Node("webcam_listener")
    {
        auto qos = rclcpp::QoS(rclcpp::SensorDataQoS());
        m_sub = image_transport::create_subscription(
            this,
            "/camera/image_raw",
            std::bind(&WebcamListener::image_callback, this, std::placeholders::_1),
            "compressed",
            qos.get_rmw_qos_profile());
        RCLCPP_INFO(this->get_logger(), "Webcam listener initialized.");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        try
        {
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::imshow("Received Webcam Stream", frame);
            int key = cv::waitKey(1);
            if (key == 27)
            {
                rclcpp::shutdown();
            }
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }

    }
    image_transport::Subscriber m_sub;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto webcam_listener_node = std::make_shared<WebcamListener>();
    rclcpp::spin(webcam_listener_node);
    rclcpp::shutdown();
    return 0;
}