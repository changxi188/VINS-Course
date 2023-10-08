#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <thread>

#include <glog/logging.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "System.h"

const int   nDelayTimes  = 2;
std::string sData_path   = "/media/cheng/holo1/rosbag/MH_05_difficult/mav0/";
std::string sConfig_path = "./config/";

std::atomic<bool> pub_running(true);

std::shared_ptr<System> pSystem;

void SignalHandler(int signum)
{
    std::cout << "Caught signal : " << signum << std::endl;
    pub_running = false;
}

void PubImuData()
{
    std::string sImu_data_file = sConfig_path + "MH_05_imu0.txt";
    LOG(INFO) << "PubImuData start sImu_data_filea: " << sImu_data_file;
    std::ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open())
    {
        LOG(ERROR) << "Failed to open imu file! " << sImu_data_file;
        return;
    }

    std::string     sImu_line;
    double          dStampNSec = 0.0;
    Eigen::Vector3d vAcc;
    Eigen::Vector3d vGyr;
    while (pub_running && std::getline(fsImu, sImu_line) && !sImu_line.empty())  // read imu data
    {
        std::istringstream ssImuData(sImu_line);
        ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
        // cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() <<
        // endl;
        pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
        usleep(5000 * nDelayTimes);
    }
    fsImu.close();
}

void PubImageData()
{
    std::string sImage_file = sConfig_path + "MH_05_cam0.txt";

    LOG(INFO) << "PubImageData start sImage_file: " << sImage_file;

    std::ifstream fsImage;
    fsImage.open(sImage_file.c_str());
    if (!fsImage.is_open())
    {
        LOG(INFO) << "Failed to open image file! " << sImage_file;
        return;
    }

    std::string sImage_line;
    double      dStampNSec;
    std::string sImgFileName;

    // cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
    while (pub_running && std::getline(fsImage, sImage_line) && !sImage_line.empty())
    {
        std::istringstream ssImuData(sImage_line);
        ssImuData >> dStampNSec >> sImgFileName;
        // cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
        std::string imagePath = sData_path + "cam0/data/" + sImgFileName;

        cv::Mat img = cv::imread(imagePath.c_str(), 0);
        if (img.empty())
        {
            std::cerr << "image is empty! path: " << imagePath;
            return;
        }
        pSystem->PubImageData(dStampNSec / 1e9, img);
        // cv::imshow("SOURCE IMAGE", img);
        // cv::waitKey(0);
        usleep(50000 * nDelayTimes);
    }
    fsImage.close();
}

int main(int argc, char** argv)
{
    signal(SIGINT, SignalHandler);

    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::INFO, "./log/info.log");

    FLAGS_alsologtostderr  = 1;
    FLAGS_minloglevel      = 0;
    FLAGS_colorlogtostderr = true;

    if (argc == 3)
    {
        /*
        LOG(WARNING) << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n"
                     << "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/";
        return -1;
        */
        sData_path   = argv[1];
        sConfig_path = argv[2];
    }

    pSystem.reset(new System(sConfig_path));

    std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);

    std::thread thd_Draw(&System::Draw, pSystem);

    std::thread thd_PubImuData(PubImuData);

    std::thread thd_PubImageData(PubImageData);

    thd_PubImuData.join();

    thd_PubImageData.join();

    pSystem->SetProcessOver();

    thd_BackEnd.join();

    thd_Draw.join();

    LOG(INFO) << "main end... see you ...";
    return 0;
}
