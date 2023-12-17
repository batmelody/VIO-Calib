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
  if (first_imu) {
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }
  if (!PreIntegrations_[FrameCount_]) {
    PreIntegrations_[FrameCount_] =
        new ImuIntegration{acc_0, gyr_0, Ba_[FrameCount_], Bg_[FrameCount_]};
  }
  if (!first_imu) {
    if (FrameCount_ != 0) {
      PreIntegrations_[FrameCount_]->push_back(dt, linear_acceleration,
                                               angular_velocity);
    }
  }
  acc_0 = linear_acceleration; // variable of estimator class
  gyr_0 = angular_velocity;    // variable of estimator class
  first_imu = false;
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

void Viocalibrate::ExtrinsicROptimizer(
    double rotation_so3[3], const std::vector<Eigen::Matrix3d> &delta_R_cam,
    const std::vector<Eigen::Matrix3d> &delta_R_imu) {
  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  loss_function = new ceres::CauchyLoss(1.0);
  ExRLocalParameterization *local_parameterization_intrinsic =
      new ExRLocalParameterization();
  problem.AddParameterBlock(rotation_so3, 3, local_parameterization_intrinsic);

  for (int i = 1; i < WINDOW_SIZE - 1; i++) {
    ceres::CostFunction *costFunction =
        new ExRFactor(delta_R_cam[i], delta_R_imu[i]);
    problem.AddResidualBlock(costFunction, loss_function, rotation_so3);
  }
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.use_nonmonotonic_steps = true;
  options.max_num_iterations = 100;
  options.function_tolerance = 1e-9;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-9;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << rotation_so3[i] << " ";
  }
}

bool Viocalibrate::CalibrateExtrinsicR(
    Eigen::Matrix3d &calib_ric_result,
    const std::vector<Eigen::Matrix3d> &delta_R_cam,
    const std::vector<Eigen::Matrix3d> &delta_R_imu) {
  ric = Eigen::Matrix3d::Identity();
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    Rc.push_back(delta_R_cam[i]);
    Rimu.push_back(delta_R_imu[i]);
  }

  Eigen::MatrixXd A(WINDOW_SIZE * 4, 4);
  A.setZero();
  int sum_ok = 0;
  for (int i = 1; i < WINDOW_SIZE - 1; i++) {
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
  Eigen::Quaterniond Qic(ric);
  Sophus::SO3d Ric_SO3(Qic);
  Eigen::Vector3d Vic = Ric_SO3.log();
  double rotation_so3[3];
  rotation_so3[0] = Vic.x();
  rotation_so3[1] = Vic.y();
  rotation_so3[2] = Vic.z();
  ExtrinsicROptimizer(rotation_so3, delta_R_cam, delta_R_imu);
  Ric_SO3 =
      Sophus::SO3d::exp({rotation_so3[0], rotation_so3[1], rotation_so3[2]});
  ric = Ric_SO3.matrix();
  return true;
}

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
  first_imu = true;
  FrameCount_ = 0;
  Imu_g = {0, 0, 9.81};
}

bool Viocalibrate::CalCulateValidation(
    std::vector<Eigen::Matrix3d> delta_R_cam,
    std::vector<Eigen::Matrix3d> delta_R_imu) {

  Eigen::Matrix3d Rbc;
  Rbc << 0, 0, 1, 1, 0, 0, 0, 1, 0;
  Eigen::AngleAxisd angle_axis(Rbc);

  double theta = angle_axis.angle();
  Eigen::Vector3d rotation_vector = angle_axis.axis();

  std::cout << "Rotation angle (in radians):\n" << theta << std::endl;
  std::cout << "Rotation vector:\n" << rotation_vector << std::endl;

  return true;
}

void Viocalibrate::ValidOptimizer() {

  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  loss_function = new ceres::CauchyLoss(1.0);

  int pos_num = 10;
  double Extrinsic[2][3];
  Extrinsic[0][0] = 0.2;
  Extrinsic[0][1] = 0.4;
  Extrinsic[0][2] = 0.1;
  Extrinsic[1][0] = -0.4;
  Extrinsic[1][1] = 0.3;
  Extrinsic[1][2] = 0.8;

  Sophus::Vector6d Xbc;
  Xbc << 0.2, 0.0, 0.1, 0.4, 0.1, 0.1;
  Sophus::Vector6d Xbc_;
  Xbc_ << Extrinsic[0][0], Extrinsic[0][1], Extrinsic[0][2], Extrinsic[1][0],
      Extrinsic[1][1], Extrinsic[1][2];

  Sophus::SE3d Tc[pos_num];
  Sophus::SE3d Tb[pos_num];
  Sophus::SE3d Tbc = Sophus::SE3d::exp(Xbc);
  Sophus::SE3d Tbc_ = Sophus::SE3d::exp(Xbc_);
  for (int pos_id = 1; pos_id < pos_num; pos_id++) {
    Sophus::Vector6d Xc;
    Xc << 0.3, 0.4, 0.2 * pos_id, 0.1, 0.1 * pos_id, 0.1;
    Tc[pos_id] = Sophus::SE3d::exp(Xc);
    Tb[pos_id] = Tbc * Tc[pos_id] * Tbc.inverse();
  }

  ExRLocalParameterization *local_parameterization_intrinsic =
      new ExRLocalParameterization();
  problem.AddParameterBlock(Extrinsic[1], 3, local_parameterization_intrinsic);

  for (int pos_id = 1; pos_id < pos_num; pos_id++) {
    ceres::CostFunction *costFunction = new ExTFactor(Tc[pos_id], Tb[pos_id]);
    problem.AddResidualBlock(costFunction, loss_function, Extrinsic[0],
                             Extrinsic[1]);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.use_nonmonotonic_steps = true;
  options.max_num_iterations = 100;
  options.function_tolerance = 1e-18;
  options.gradient_tolerance = 1e-18;
  options.parameter_tolerance = 1e-18;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  std::cout << " Extrinsic " << Extrinsic[0][0] << " " << Extrinsic[0][1] << " "
            << Extrinsic[0][2] << " " << Extrinsic[1][0] << " "
            << Extrinsic[1][1] << " " << Extrinsic[1][2] << std::endl;
}