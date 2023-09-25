#include "parameters.h"
#include "utility/utility.h"
#include "viocali_camera_factor.h"
#include "viocali_imu_factor.h"
#include <ceres/ceres.h>
#include <condition_variable>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stdio.h>
#include <thread>

class Viocalibrate {
public:
  Viocalibrate(std::string cameraName, Camera::ModelType modelType,
               cv::Size boardSize, cv::Size imageSize, float squareSize)
      : cameraName_(cameraName), modelType_(modelType), boardSize_(boardSize),
        imageSize_(imageSize), squareSize_(squareSize) {
    CameraCalibration_ = new CameraCalibration(
        modelType_, cameraName_, imageSize_, boardSize_, squareSize_);
    InitState();
  };
  ~Viocalibrate(){};
  void IMULocalization(double dt, const Eigen::Vector3d &linear_acceleration,
                       const Eigen::Vector3d &angular_velocity);

  bool CameraLocalization(std::vector<std::vector<cv::Point3f>> world_corner,
                          std::vector<std::vector<cv::Point2f>> image_corner);

  bool CalibrateExtrinsicR(std::vector<Eigen::Matrix3d> delta_R_cam,
                           std::vector<Eigen::Matrix3d> delta_R_imu,
                           Eigen::Matrix3d &calib_ric_result);

  void SolveDeltaRFromse3(std::vector<Eigen::Matrix3d> &Rcw,
                          std::vector<Eigen::Matrix3d> &delta_R_cam);

  void InitState();

  cv::Mat GetCameraPoses(void);

  std::vector<Eigen::Matrix3d> GetCamRotation(void);

  std::vector<Eigen::Matrix3d> GetImuRotation(void);

  /*camera localization*/
  std::string cameraName_;
  Camera::ModelType modelType_;
  cv::Size boardSize_;
  cv::Size imageSize_;
  cv::Mat cameraPoses_;
  CameraCalibration *CameraCalibration_;
  float squareSize_;

  /*Imu localization*/
  ImuIntegration *PreIntegrations_[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Ba_[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Bg_[(WINDOW_SIZE + 1)];
  Eigen::Vector3d g_;
  Eigen::Vector3d acc_0;
  Eigen::Vector3d gyr_0;
  Eigen::Vector3d Imu_g;
  bool init_feature;
  bool init_imu;
  double last_imu_t;
  double latest_time;
  double td;

  /*Extrinsic Calibration*/
  std::vector<Eigen::Matrix3d> Rc;
  std::vector<Eigen::Matrix3d> Rimu;
  std::vector<Eigen::Matrix3d> Rc_g;
  Eigen::Matrix3d ric;

  /*relate to thread*/
  std::condition_variable con;
  double current_time;
  int sum_of_wait;
  bool first_imu;
  int FrameCount_;
};