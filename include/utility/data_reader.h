#pragma once
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

std::queue<ImuData> ReadImuDataFromSimulation(const std::string &filename) {
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
