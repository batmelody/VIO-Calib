#include "viocali_lidar_factor.h"

bool LidarCostFunction::Evaluate(double const *const *parameters,
                                 double *residuals, double **jacobians) const {

  Eigen::Map<Eigen::Vector3d> residual(residuals);
  Eigen::Vector3d tb0bk, tbl;
  Sophus::Vector3d theta_b0bk, theta_bl;

  // SE3 Tb0bk 3x4
  theta_b0bk << parameters[0][0], parameters[0][1], parameters[0][2];
  tb0bk << parameters[1][0], parameters[1][1], parameters[1][2];
  Sophus::SO3d Rb0bk_SO3 = Sophus::SO3d::exp(theta_b0bk);
  Eigen::Matrix3d Rb0bk = Rb0bk_SO3.matrix();

  // SE3 Tbl 3x4
  theta_bl << parameters[2][0], parameters[2][1], parameters[2][2];
  tbl << parameters[3][0], parameters[3][1], parameters[3][2];
  Sophus::SO3d Rbl_SO3 = Sophus::SO3d::exp(theta_bl);
  Eigen::Matrix3d Rbl = Rbl_SO3.matrix();

  // body point
  Eigen::Vector3d body_point = Rbl * lidar_point_ + tbl;
  Eigen::Matrix3d body_point_cross_mat = Utility::skewSymmetric(body_point);
  Eigen::Matrix3d lidar_point_cross_mat = Utility::skewSymmetric(lidar_point_);

  residuals[0] = plane_normal_.transpose() *
                     (Rbl.transpose() * Rb0bk * (Rbl * lidar_point_ + tbl) +
                      Rbl.transpose() * tb0bk - Rbl.transpose() * tbl) -
                 plane_distance_;

  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_Rb0bk(
          jacobians[0]);
      jacobian_Rb0bk.setZero();
      jacobian_Rb0bk = plane_normal_.transpose() *
                       (-Rbl.transpose() * Rb0bk *
                        Utility::skewSymmetric(Rbl * lidar_point_ + tbl));
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_tb0bk(
          jacobians[1]);
      jacobian_tb0bk.setZero();
      jacobian_tb0bk = plane_normal_.transpose() *
                       (Rbl.transpose() * Eigen::Matrix3d::Identity());
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_Rbl(
          jacobians[2]);
      jacobian_Rbl.setZero();
      jacobian_Rbl = plane_normal_.transpose() *
                     (-Rbl.transpose() * Rb0bk * Rbl *
                          Utility::skewSymmetric(lidar_point_) +
                      Utility::skewSymmetric(Rbl.transpose() * Rb0bk * Rbl *
                                             lidar_point_) +
                      Utility::skewSymmetric(Rbl.transpose() * Rb0bk * tb0bk) +
                      Utility::skewSymmetric(Rbl.transpose() * tb0bk) -
                      Utility::skewSymmetric(Rbl.transpose() * tbl));
    }
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_tbl(
          jacobians[3]);
      jacobian_tbl.setZero();
      jacobian_tbl = plane_normal_.transpose() *
                     (Rbl.transpose() * Rb0bk - Rbl.transpose());
    }
  }
  return true;
}

bool LidarLocalParam::Plus(const double *x, const double *delta,
                           double *x_plus_delta) const {
  const Eigen::Map<const Eigen::Matrix<double, 3, 1>> phi(x);
  const Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_phi(delta);
  Eigen::Map<Eigen::Matrix<double, 3, 1>> x_plus_delta_phi(x_plus_delta);
  Sophus::SO3d R = Sophus::SO3d::exp(phi);
  Sophus::SO3d delta_R = Sophus::SO3d::exp(delta_phi);
  x_plus_delta_phi = (R * delta_R).log();
  return true;
}
