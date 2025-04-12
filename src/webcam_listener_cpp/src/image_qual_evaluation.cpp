#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <optional>
#include <cmath>

/**
 * Subscribes to: Raw and Compressed Images
 */

class ImageQualityEvaluator: public rclcpp::Node
{
public:
    ImageQualityEvaluator():
        Node("image_quality_evaluator")
    {
        auto qos = rclcpp::QoS(rclcpp::SensorDataQoS());
        m_raw_sub = image_transport::create_subscription(
            this,
            "/camera/image_raw",
            std::bind(&ImageQualityEvaluator::raw_image_callback, this, std::placeholders::_1),
            "raw",
            qos.get_rmw_qos_profile());

        m_compr_sub = image_transport::create_subscription(
            this,
            "/camera/image_raw",
            std::bind(&ImageQualityEvaluator::compr_image_callback, this, std::placeholders::_1),
            "compressed",
            qos.get_rmw_qos_profile());
        
        RCLCPP_INFO(this->get_logger(), "ImageQualityEvaluator initialized.");
        
    }

private:
    using TimeStampedImage = std::pair<rclcpp::Time, cv::Mat>;

    void raw_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        rclcpp::Time raw_time(msg->header.stamp);
        RCLCPP_INFO(this->get_logger(), "Received Raw Image at: %.3f sec", raw_time.seconds());
        cv::Mat raw_img;
        try
        {
            raw_img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Raw cv_bridge exception: %s", e.what());
            return;
        }

        auto compr_image = find_closest_compressed_match(raw_time);
        if(compr_image.has_value())
        {
            process_pair(raw_img, compr_image.value());
        }
    }

    void compr_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        rclcpp::Time compressed_img_time(msg->header.stamp);
        RCLCPP_INFO(this->get_logger(), "Received Compressed Image at: %.3f sec", compressed_img_time.seconds());
        cv::Mat compr_img;
        try
        {
            compr_img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
            m_compressed_buffer.emplace_back(compressed_img_time, compr_img);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Raw cv_bridge exception: %s", e.what());
            return;
        }

        while(m_compressed_buffer.size() > MAX_BUFFER_SIZE)
        {
            m_compressed_buffer.pop_front();
        }
    }

    std::optional<cv::Mat> find_closest_compressed_match(const rclcpp::Time& raw_time)
    {
        if(m_compressed_buffer.empty())
        {
            return std::nullopt;
        }

        for(const auto&[compr_time, img]: m_compressed_buffer)
        {
            double diff = std::abs((compr_time - raw_time).seconds());
            if(diff <= TIME_OFFSET_SECONDS)
            {
                return img;
            }
        }
        return std::nullopt;
    }

    void process_pair(const cv::Mat& raw_img, const cv::Mat& compressed_img)
    {
        RCLCPP_INFO(this->get_logger(), "Processing matched frame pair...");
        // Testing to see if psnr drops
        // cv::Mat noisy_compressed = compressed_img.clone();
        // noisy_compressed += cv::Scalar(10, 10, 10);
        double psnr = getPSNR(raw_img, compressed_img);
        RCLCPP_INFO(this->get_logger(), "PSNR %.3f dB", psnr);
        cv::Scalar mssim = getMSSIM(raw_img, compressed_img);
        RCLCPP_INFO(this->get_logger(), "SSIM: B=%.4f, G=%.4f, R=%.4f",
                    mssim[0], mssim[1], mssim[2]);
    }

    double getPSNR(const cv::Mat& raw_img, const cv::Mat& compressed_img)
    {
        cv::Mat diff;
        cv::absdiff(raw_img, compressed_img, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);
        // this will be 4D array (Scalar is array)...sum at each channel.
        cv::Scalar sum_sqred_diff = cv::sum(diff);
        double total_quared_err = sum_sqred_diff[0] + sum_sqred_diff[1] + sum_sqred_diff[2];
        if( total_quared_err <= 1e-10)
            return 0;
        
        double MSE = total_quared_err / (double)(raw_img.channels()*raw_img.total());
        double PSNR = 10 * std::log10((255*255)/MSE);
        return PSNR;
    }

    cv::Scalar getMSSIM(const cv::Mat& raw_img, const cv::Mat& compressed_img)
    {
        // Refer for original SSIM theory: https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
        // Code from: https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html
        // And this: https://arxiv.org/pdf/2006.13846
        const double C1 = 6.5025, C2 = 58.5225;
        cv::Mat I1, I2;
        raw_img.convertTo(I1, CV_32F);
        compressed_img.convertTo(I2, CV_32F);
     
        cv::Mat I2_2   = I2.mul(I2);
        cv::Mat I1_2   = I1.mul(I1);
        cv::Mat I1_I2  = I1.mul(I2);
     
        cv::Mat mu1, mu2;
        // Gaussian blur is used to calculate patchwise mean.
        cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
        cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
     
        cv::Mat mu1_2   = mu1.mul(mu1);
        cv::Mat mu2_2   = mu2.mul(mu2);
        cv::Mat mu1_mu2 = mu1.mul(mu2);
     
        cv::Mat sigma1_2, sigma2_2, sigma12;
     
        cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
     
        cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
     
        cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
     
        cv::Mat t1, t2, t3;
     
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);
     
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);
     
        cv::Mat ssim_map;
        cv::divide(t3, t1, ssim_map);
     
        cv::Scalar mssim = cv::mean(ssim_map);
        return mssim;
    }

    std::deque<TimeStampedImage> m_compressed_buffer;
    const size_t MAX_BUFFER_SIZE = 10;
    const double TIME_OFFSET_SECONDS = 0.010; // 10mS

    image_transport::Subscriber m_raw_sub;
    image_transport::Subscriber m_compr_sub;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto image_qual_eval_node = std::make_shared<ImageQualityEvaluator>();
    rclcpp::spin(image_qual_eval_node);
    rclcpp::shutdown();
    return 0;
}