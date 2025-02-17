#pragma once
#include "include/utility/data_reader.h"
#include "viocali_calibrate.h"
#include <Eigen/Dense> // 包含Eigen库
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

std::string SAVE_DIR = "../data/pose_pre_ini_imu.txt";

std::vector<std::pair<std::vector<ImuData>, CameraData>>
DataSynthesis(std::queue<ImuData> imu_buffer, std::queue<CameraData> cam_buffer,
              double td) {
  std::vector<std::pair<std::vector<ImuData>, CameraData>> measurements;
  int sum_of_wait = 0;
  while (true) {
    if (imu_buffer.empty() || cam_buffer.empty())
      return measurements;
    if (!(imu_buffer.back().timestamp > cam_buffer.front().timestamp + td)) {
      // ROS_WARN("wait for imu, only should happen at the beginning");
      sum_of_wait++;
      return measurements;
    }
    if (!(imu_buffer.front().timestamp < cam_buffer.front().timestamp + td)) {
      printf("throw img, only should happen at the beginning");
      cam_buffer.pop();
      continue;
    }
    CameraData CAM = cam_buffer.front();
    cam_buffer.pop();
    std::vector<ImuData> IMUs;
    while (imu_buffer.front().timestamp < CAM.timestamp + td) {
      IMUs.emplace_back(imu_buffer.front());
      imu_buffer.pop();
    }
    IMUs.emplace_back(imu_buffer.front());
    if (IMUs.empty())
      printf("no imu between two image");
    measurements.emplace_back(IMUs, CAM);
  }
  return measurements;
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
      // ROS_WARN("wait for imu, only should happen at the beginning");
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

int RunForRealData() {
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
  viocali->SolveCamDeltaR(Rcw, delta_R_cam);
  delta_R_imu = viocali->GetImuRotation();
  if (viocali->CalibrateExtrinsicR(viocali->ric, delta_R_cam, delta_R_imu)) {
    std::cout << " success " << std::endl;
  } else {
    std::cout << "false " << std::endl;
  }
  delete viocali;
  return 0;
}

int GenerateEucmGT(int row, int col, double alpha, double beta,
                   Sophus::Vector6d Xi, double chessboardSize,
                   std::vector<Eigen::Vector3d> &Pw,
                   std::vector<Eigen::Vector3d> &Pc,
                   std::vector<Eigen::Vector3d> &Pm,
                   std::vector<Eigen::Vector3d> &p) {

  Sophus::SE3d T = Sophus::SE3d::exp(Xi);

  for (int i = -row; i < row + 1; i++) {
    for (int j = -col; j < col + 1; j++) {
      Pw.push_back(Eigen::Vector3d(i * chessboardSize, j * chessboardSize, 0));
    }
  }

  for (int i = 0; i < Pw.size(); i++) {
    Pc.push_back(T * Pw[i]);
  }

  for (int i = 0; i < Pc.size(); i++) {
    double x = Pc[i].x();
    double y = Pc[i].y();
    double z = Pc[i].z();
    double rho = sqrt(beta * (x * x + y * y) + z * z);
    Pm.push_back(Eigen::Vector3d(x / (alpha * rho + (1 - alpha) * z),
                                 y / (alpha * rho + (1 - alpha) * z), 1));
  }
  double fx = 460;
  double fy = 460;
  double cx = 320;
  double cy = 240;
  for (int i = 0; i < Pm.size(); i++) {
    p.push_back(Eigen::Vector3d(fx * Pm[i].x() + cx, fy * Pm[i].y() + cy, 1));
  }
  return 0;
}

int EucmSimulation() {
  int pos_num = 10;
  double alpha = 0.571;
  double beta = 1.18;
  int row = 20;
  int col = 20;
  double chessboardSize = 0.2;
  double para_Pose[pos_num][6];
  std::vector<Eigen::Vector3d> Pw[pos_num];
  std::vector<Eigen::Vector3d> Pm[pos_num];
  std::vector<Eigen::Vector3d> Pc[pos_num];
  std::vector<Eigen::Vector3d> p[pos_num];
  Sophus::Vector6d Xi[pos_num];

  for (int pos_id = 0; pos_id < pos_num; pos_id++) {
    double left_limit = (-pos_num * 0.1) / 2;
    double right_limit = (pos_num * 0.1) / 2;
    Xi[pos_id] << 0.0, 0.0, 0.1 * pos_id, right_limit - 0.1 * pos_id,
        left_limit + 0.1 * pos_id, 0.0;
    GenerateEucmGT(row, col, alpha, beta, Xi[pos_id], chessboardSize,
                   Pw[pos_id], Pc[pos_id], Pm[pos_id], p[pos_id]);
  }
  cv::Size boardSize(20, 20);
  cv::Size imageSize(640, 480);
  float squareSize = 0.2;
  int WindowSize = 20;
  std::string cameraName = "cam_name";
  Camera::ModelType modelType = Camera::ModelType::EUCM;
  std::vector<std::vector<cv::Point3f>> world_corner;
  std::vector<std::vector<cv::Point2f>> image_corner;
  std::vector<cv::Point3f> objs;
  std::vector<cv::Point2f> cornerses;
  for (int pos_id = 0; pos_id < pos_num; pos_id++) {
    std::vector<cv::Point3f> objs;
    std::vector<cv::Point2f> cornerses;
    for (int i = 0; i < p[pos_id].size(); i++) {
      if (p[pos_id][i].x() > 1e-6 && p[pos_id][i].y() > 1e-6 &&
          Pm[pos_id][i].z() > 1e-6) {
        cv::Point3f obj;
        cv::Point2f corners;
        obj.x = static_cast<float>(Pw[pos_id][i].x());
        obj.y = static_cast<float>(Pw[pos_id][i].y());
        obj.z = static_cast<float>(Pw[pos_id][i].z());

        corners.x = static_cast<float>(p[pos_id][i].x());
        corners.y = static_cast<float>(p[pos_id][i].y());
        objs.push_back(obj);
        cornerses.push_back(corners);
      }
    }
    world_corner.push_back(objs);
    image_corner.push_back(cornerses);
  }
  Viocalibrate *viocali =
      new Viocalibrate(cameraName, modelType, boardSize, imageSize, squareSize);

  bool isCamPose = viocali->CameraLocalization(world_corner, image_corner);
  cv::Mat CamPose = viocali->GetCameraPoses();
  delete viocali;
  return 0;
}

int RunForSynthesisData() {
  /*init parameters*/
  cv::FileStorage fs("../config/vio-config.yaml", cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open config file!" << std::endl;
    return -1;
  }
  cv::Size boardSize;
  cv::Size imageSize;
  float squareSize;
  int WindowSize = 20;
  double current_time = -0.01;
  std::string cameraName;
  std::string imuFileName; // 替换为你的CSV文件名
  std::string camFileName; // 替换为你的CSV文件名

  fs["boardsize"] >> boardSize;
  fs["imagesize"] >> imageSize;
  fs["squaresize"] >> squareSize;
  fs["cam_name"] >> cameraName;
  fs["imu_dir"] >> imuFileName;
  fs["cam_dir"] >> camFileName;

  Camera::ModelType modelType = Camera::ModelType::PINHOLE;
  std::vector<std::vector<cv::Point3f>> world_corner;
  std::vector<std::vector<cv::Point2f>> image_corner;
  std::vector<cv::Point3f> obj;
  std::vector<cv::Point2f> corners;
  std::vector<Eigen::Quaterniond> Qwcs;
  std::vector<Eigen::Matrix3d> delta_R_cam;
  std::vector<Eigen::Matrix3d> delta_R_imu;

  auto imuData = ReadImuDataFromSimulation(imuFileName);
  auto camData = ReadCamDataFromSimulation(camFileName);
  std::vector<std::pair<std::vector<ImuData>, CameraData>> Mes =
      DataSynthesis(imuData, camData, 0); //仿真数据暂时不设置td
  double imu_first, cam_first;
  imu_first = (Mes[0].first[0].timestamp);
  cam_first = (Mes[0].second.timestamp);
  Viocalibrate *viocali =
      new Viocalibrate(cameraName, modelType, boardSize, imageSize, squareSize);
  std::ofstream save_points;
  save_points.open(SAVE_DIR);
  for (auto &mes : Mes) {
    for (auto &imu : mes.first) {
      double imu_t = imu.timestamp;
      double cam_t = mes.second.timestamp;
      if (imu_t <= cam_t) {
        double dt = imu_t - current_time;
        current_time = imu_t;
        viocali->IMULocalization(dt, imu.acc, imu.gyro);
      } else {
        double dt_1 = cam_t - current_time;
        double dt_2 = imu_t - cam_t;
        current_time = cam_t;
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        double dx = w1 * dx + w2 * imu.acc.x();
        double dy = w1 * dy + w2 * imu.acc.y();
        double dz = w1 * dz + w2 * imu.acc.z();
        double rx = w1 * rx + w2 * imu.gyro.x();
        double ry = w1 * ry + w2 * imu.gyro.y();
        double rz = w1 * rz + w2 * imu.gyro.z();
        viocali->IMULocalization(dt_1, Eigen::Vector3d(dx, dy, dz),
                                 Eigen::Vector3d(rx, ry, rz));
      }
    }
    Qwcs.push_back(mes.second.Qwc);
    if (viocali->FrameCount_ == WINDOW_SIZE) {
      break;
    } else {
      viocali->FrameCount_++;
    }
  }
  std::vector<Eigen::Matrix3d> Rwc;
  for (auto &q : Qwcs) {
    Rwc.push_back(q.toRotationMatrix());
  }
  viocali->SolveCamDeltaR(Rwc, delta_R_cam);
  delta_R_imu.clear();
  delta_R_imu = viocali->GetImuRotation();
  Eigen::Matrix3d Ric;
  viocali->CalibrateExtrinsicR(viocali->ric, delta_R_cam, delta_R_imu);
  delete viocali;
  return 0;
}

int main() {
  RunForRealData();
  EucmSimulation();
}