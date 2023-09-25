#pragma once
#include "viocali_calibrate.h"
#include <Eigen/Dense> // 包含Eigen库
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

struct ImuData {
  double timestamp;
  Eigen::Vector3d gyro;
  Eigen::Vector3d acc;
};

struct ImageData {
  bool isFound;
  double timestamp;
  std::vector<cv::Point2f> Points2d;
  std::vector<cv::Point3f> Points3d;
};

std::queue<ImuData> ReadImuData(const std::string &filename) {
  int unit_time = 1000000000;
  std::queue<ImuData> data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return data;
  }
  std::string line;
  while (std::getline(file, line)) {
    ImuData imuData;
    int col = 0;
    std::stringstream ss(line);
    std::string str;
    while (std::getline(ss, str, ',')) {
      if (col == 0) {
        imuData.timestamp = std::stod(str) / unit_time;
      }
      if (col > 0 & col < 4) {
        imuData.gyro[col - 1] = std::stod(str);
      }
      if (col > 3) {
        imuData.acc[col - 4] = std::stod(str);
      }
      col++;
    }
    data.push(imuData);
  }
  file.close();
  return data;
}

std::queue<ImageData> ReadImgData(const std::string &path,
                                  const cv::Size PatternSize,
                                  const float BoardSize) {
  int unit_time = 1000000000;
  std::vector<std::string> filenames;
  cv::glob(path, filenames);
  std::queue<ImageData> images_data;
  std::vector<cv::Point2f> corners;
  std::vector<cv::Point3f> obj;
  int cnt = 0;
  for (const auto &filename : filenames) {
    cnt++;
    ImageData imgData;
    imgData.Points3d.clear();
    imgData.Points2d.clear();
    obj.clear();
    corners.clear();
    std::string imgname = filename.substr(filename.find_last_of("/\\") + 1);
    imgname = imgname.substr(0, imgname.find_last_of("."));
    double timestamp = std::stod(imgname) / unit_time;

    cv::Mat image = cv::imread(filename);
    bool found = cv::findChessboardCorners(
        image, PatternSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
            cv::CALIB_CB_FILTER_QUADS + cv::CALIB_CB_FAST_CHECK);
    for (int i = 0; i < PatternSize.height; ++i) {
      for (int j = 0; j < PatternSize.width; ++j) {
        obj.emplace_back(j * BoardSize, i * BoardSize, 0);
      }
    }
    if (!found) {
      break;
    }
    imgData.isFound = found;
    imgData.timestamp = timestamp;
    imgData.Points3d = obj;
    imgData.Points2d = corners;
    images_data.push(imgData);
  }
  return images_data;
}

std::vector<std::pair<std::vector<ImuData>, ImageData>>
DataSynthesis(std::queue<ImuData> imu_buffer, std::queue<ImageData> img_buffer,
              double td) {
  std::vector<std::pair<std::vector<ImuData>, ImageData>> measurements;
  int sum_of_wait = 0;
  while (true) {
    if (imu_buffer.empty() || img_buffer.empty())
      return measurements;
    if (!(imu_buffer.back().timestamp > img_buffer.front().timestamp + td)) {
      sum_of_wait++;
      return measurements;
    }
    if (!(imu_buffer.front().timestamp < img_buffer.front().timestamp + td)) {
      printf("throw img, only should happen at the beginning");
      img_buffer.pop();
      continue;
    }
    ImageData IMG = img_buffer.front();
    img_buffer.pop();
    std::vector<ImuData> IMUs;
    while (imu_buffer.front().timestamp < IMG.timestamp + td) {
      IMUs.emplace_back(imu_buffer.front());
      imu_buffer.pop();
    }
    IMUs.emplace_back(imu_buffer.front());
    if (IMUs.empty())
      printf("no imu between two image");
    measurements.emplace_back(IMUs, IMG);
  }
  return measurements;
}

int main() {
  const std::string imufilename =
      "/home/weizhengyu/Data/rs/imu0.csv"; // 替换为你的CSV文件名
  const std::string imgfilename =
      "/home/weizhengyu/Data/rs/cam0/"; // 替换为你的CSV文件名

  cv::Size boardSize(11, 16);
  cv::Size imageSize(1280, 960);
  float squareSize = 0.04;
  int WindowSize = 20;
  double current_time = -1;

  auto imuData = ReadImuData(imufilename);
  std::cout << "imuDataSize: " << imuData.size() << std::endl;
  auto imgData = ReadImgData(imgfilename, boardSize, squareSize);
  std::cout << "imgDataSize: " << imgData.size() << std::endl;

  std::string cameraName = "WZY Camera";
  Camera::ModelType modelType = Camera::ModelType::PINHOLE;
  std::vector<std::vector<cv::Point3f>> world_corner;
  std::vector<std::vector<cv::Point2f>> image_corner;
  std::vector<cv::Point3f> obj;
  std::vector<cv::Point2f> corners;
  std::vector<Eigen::Matrix3d> delta_R_cam;
  std::vector<Eigen::Matrix3d> delta_R_imu;

  std::vector<std::pair<std::vector<ImuData>, ImageData>> Mes =
      DataSynthesis(imuData, imgData, 0.001);
  double imu_first, img_first;
  imu_first = (Mes[0].first[0].timestamp);
  img_first = (Mes[0].second.timestamp);

  Viocalibrate *viocali =
      new Viocalibrate(cameraName, modelType, boardSize, imageSize, squareSize);

  for (auto &mes : Mes) {
    for (auto &imu : mes.first) {
      double imu_t = imu.timestamp;
      double img_t = mes.second.timestamp;
      if (imu_t <= img_t) {
        if (current_time < 0)
          current_time = imu_t;
        double dt = imu_t - current_time;
        current_time = imu_t;
        std::cout << " dt: " << dt << std::endl;
        viocali->IMULocalization(dt, imu.acc, imu.gyro);
      } else {
        double dt_1 = img_t - current_time;
        double dt_2 = imu_t - img_t;
        current_time = img_t;
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        double dx = w1 * dx + w2 * imu.acc.x();
        double dy = w1 * dy + w2 * imu.acc.y();
        double dz = w1 * dz + w2 * imu.acc.z();
        double rx = w1 * rx + w2 * imu.gyro.x();
        double ry = w1 * ry + w2 * imu.gyro.y();
        double rz = w1 * rz + w2 * imu.gyro.z();
        std::cout << "dt_1: " << dt_1 << std::endl;
        viocali->IMULocalization(dt_1, Eigen::Vector3d(dx, dy, dz),
                                 Eigen::Vector3d(rx, ry, rz));
      }
    }
    world_corner.push_back(mes.second.Points3d);
    image_corner.push_back(mes.second.Points2d);
    if (viocali->FrameCount_ == WINDOW_SIZE) {
      break;
    } else {
      viocali->FrameCount_++;
    }
    std::cout << "FrameCount_: " << viocali->FrameCount_ << std::endl;
  }

  bool isCamPose = viocali->CameraLocalization(world_corner, image_corner);
  cv::Mat CamPose = viocali->GetCameraPoses();
  std::vector<Eigen::Matrix3d> Rcw = viocali->GetCamRotation();
  viocali->SolveDeltaRFromse3(Rcw, delta_R_cam);
  delta_R_imu = viocali->GetImuRotation();

  if (viocali->CalibrateExtrinsicR(delta_R_cam, delta_R_imu, viocali->ric)) {
    std::cout << " success " << std::endl;
  } else {
    std::cout << "false " << std::endl;
  }
  delete viocali;
  return 0;
}
