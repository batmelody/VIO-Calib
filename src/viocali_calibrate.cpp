#include "viocali_calibrate.h"

cv::Mat Viocalibrate::GetCameraPoses(void) { return cameraPoses_; };

std::vector<Eigen::Matrix3d> Viocalibrate::GetCamRotation(void) {
  std::vector<Eigen::Matrix3d> Rcam;
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    cv::Mat rvec(3, 1, CV_64F);
    rvec.at<double>(0) = cameraPoses_.at<double>(i, 0);
    rvec.at<double>(1) = cameraPoses_.at<double>(i, 1);
    rvec.at<double>(2) = cameraPoses_.at<double>(i, 2);
    cv::Mat R0;
    cv::Rodrigues(rvec, R0);
    Eigen::Matrix3d R(3, 3);
    R << R0.at<double>(0, 0), R0.at<double>(0, 1), R0.at<double>(0, 2),
        R0.at<double>(1, 0), R0.at<double>(1, 1), R0.at<double>(1, 2),
        R0.at<double>(2, 0), R0.at<double>(2, 1), R0.at<double>(2, 2);
    Rcam.push_back(R);
  }
  return Rcam;
};

std::vector<Eigen::Matrix3d> Viocalibrate::GetImuRotation(void) {
  std::vector<Eigen::Matrix3d> Rimu;
  for (int i = 1; i < WINDOW_SIZE + 1; i++) {
    Rimu.push_back(PreIntegrations_[i]->delta_q.toRotationMatrix());
  }
  return Rimu;
};

void Viocalibrate::IMULocalization(double dt,
                                   const Eigen::Vector3d &linear_acceleration,
                                   const Eigen::Vector3d &angular_velocity) {
  std::cout << "function FrameCount_: " << FrameCount_ << std::endl;
  Ba_[FrameCount_].setZero();
  Bg_[FrameCount_].setZero();
  if (!first_imu) {
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }
  if (!PreIntegrations_[FrameCount_]) {
    std::cout << "no thing" << std::endl;
  } else {
    std::cout << " some thing" << std::endl;
  }

  if (!PreIntegrations_[FrameCount_]) {
    std::cout << "in create if" << std::endl;
    PreIntegrations_[FrameCount_] =
        new ImuIntegration{acc_0, gyr_0, Ba_[FrameCount_], Bg_[FrameCount_]};
  }
  if (FrameCount_ != 0) {
    std::cout << "in push if" << std::endl;
    PreIntegrations_[FrameCount_]->push_back(dt, linear_acceleration,
                                             angular_velocity);
  }
  std::cout << " IMULocalization " << std::endl;
  acc_0 = linear_acceleration; // variable of estimator class
  gyr_0 = angular_velocity;    // variable of estimator class
}

bool Viocalibrate::CameraLocalization(
    std::vector<std::vector<cv::Point3f>> world_corner,
    std::vector<std::vector<cv::Point2f>> image_corner) {
  if (!CameraCalibration_) {
    std::cout << "_____There is No CameraCalibrator______" << std::endl;
    return false;
  }
  CameraCalibration_->addChessboardData(world_corner, image_corner);
  CameraCalibration_->calibrate();
  cameraPoses_ = CameraCalibration_->cameraPoses();
  return true;
}

bool Viocalibrate::CalibrateExtrinsicR(std::vector<Eigen::Matrix3d> delta_R_cam,
                                       std::vector<Eigen::Matrix3d> delta_R_imu,
                                       Eigen::Matrix3d &calib_ric_result) {
  ric = Eigen::Matrix3d::Identity();
  for (int i = 0; i < WINDOW_SIZE; i++) {
    Rc.push_back(delta_R_cam[i]);
    Rimu.push_back(delta_R_imu[i]);
    Rc_g.push_back(ric.inverse() * delta_R_imu[i] * ric);
  }

  Eigen::MatrixXd A(WINDOW_SIZE * 4, 4);
  A.setZero();
  int sum_ok = 0;
  for (int i = 1; i <= WINDOW_SIZE; i++) {
    Eigen::Quaterniond r1(Rc[i]);
    Eigen::Quaterniond r2(Rc_g[i]);

    double angular_distance = 180 / M_PI * r1.angularDistance(r2);
    double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
    ++sum_ok;
    Eigen::Matrix4d L, R;
    double w = Eigen::Quaterniond(Rc[i]).w();
    Eigen::Vector3d q = Eigen::Quaterniond(Rc[i]).vec();
    L.block<3, 3>(0, 0) =
        w * Eigen::Matrix3d::Identity() + Utility::skewSymmetric(q);
    L.block<3, 1>(0, 3) = q;
    L.block<1, 3>(3, 0) = -q.transpose();
    L(3, 3) = w;

    Eigen::Quaterniond R_ij(Rimu[i]);
    w = R_ij.w();
    q = R_ij.vec();
    R.block<3, 3>(0, 0) =
        w * Eigen::Matrix3d::Identity() - Utility::skewSymmetric(q);
    R.block<3, 1>(0, 3) = q;
    R.block<1, 3>(3, 0) = -q.transpose();
    R(3, 3) = w;

    A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
  Eigen::Quaterniond estimated_R(x);
  ric = estimated_R.toRotationMatrix().inverse();
  // std::cout << svd.singularValues().transpose() << std::endl;
  // std::cout << ric << std::endl;
  Eigen::Vector3d ric_cov;
  ric_cov = svd.singularValues().tail<3>();
  if (ric_cov(1) > 0.25) {
    calib_ric_result = ric;
    std::cout << "calib_ric_result: " << calib_ric_result << std::endl;

    return true;
  } else {
    std::cout << "calib_ric_result: " << calib_ric_result << std::endl;
    return false;
  }
}

void Viocalibrate::SolveDeltaRFromse3(
    std::vector<Eigen::Matrix3d> &Rcw,
    std::vector<Eigen::Matrix3d> &delta_R_cam) {
  for (int i = 0; i < WINDOW_SIZE; i++) {
    Eigen::Matrix3d Rcur = Rcw[i];
    Eigen::Matrix3d Rnext = Rcw[i + 1];
    Rnext = Rnext.inverse().eval();
    delta_R_cam.push_back(Rcur * Rnext);
  }
}

void Viocalibrate::InitState() {
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    PreIntegrations_[i] = nullptr;
  }
  init_imu = 1;
  last_imu_t = 0;
  latest_time;
  td = 0.0006;
  current_time = -1;
  sum_of_wait = 0;
  first_imu = false;
  FrameCount_ = 0;
  Imu_g = {0, 0, 9.81};
};
