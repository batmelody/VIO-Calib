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
  Ba_[FrameCount_].setZero();
  Bg_[FrameCount_].setZero();
  if (!first_imu) {
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }
  if (!PreIntegrations_[FrameCount_]) {
    PreIntegrations_[FrameCount_] =
        new ImuIntegration{acc_0, gyr_0, Ba_[FrameCount_], Bg_[FrameCount_]};
  }
  if (FrameCount_ != 0) {
    PreIntegrations_[FrameCount_]->push_back(dt, linear_acceleration,
                                             angular_velocity);
  }
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
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    Rc.push_back(delta_R_cam[i]);
    Rimu.push_back(delta_R_imu[i]);
  }

  Eigen::MatrixXd A(2 * 4, 4);
  A.setZero();
  int sum_ok = 0;
  for (int i = 1; i <= 2; i++) {
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

    A.block<4, 4>((i - 1) * 4, 0) = (L - R);
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
  Eigen::Quaterniond estimated_R(x);
  ric = estimated_R.toRotationMatrix().inverse();
  Eigen::Vector3d ric_cov;
  ric_cov = svd.singularValues().tail<3>();
  std::cout << "ric: " << std::endl << ric;
  if (ric_cov(1) > 0.25) {
    calib_ric_result = ric;
    return true;
  } else {
    return false;
  }
}

/*Note: the rotations are from camera-frame to world-frame*/
void Viocalibrate::SolveCamDeltaR(std::vector<Eigen::Matrix3d> &Rwc,
                                  std::vector<Eigen::Matrix3d> &delta_R_cam) {
  for (int i = 0; i < WINDOW_SIZE; i++) {
    Eigen::Matrix3d Rcur = Rwc[i];
    Eigen::Matrix3d Rnext = Rwc[i + 1];
    Rcur = Rcur.inverse().eval();
    delta_R_cam.push_back((Rcur * Rnext));
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
}

bool Viocalibrate::ExtrinsicValidation(std::vector<Eigen::Matrix3d> delta_R_cam,
                                       std::vector<Eigen::Matrix3d> delta_R_imu,
                                       Eigen::Matrix3d &calib_ric_result) {

  for (int i = 0; i < delta_R_cam.size(); i++) {
    std::cout << "imu: " << std::endl
              << delta_R_imu[i] * calib_ric_result << std::endl;
    std::cout << "cam: " << std::endl
              << calib_ric_result * delta_R_cam[i] << std::endl;
  }
  std::cout << "ric: " << std::endl << calib_ric_result << std::endl;
  return true;
}