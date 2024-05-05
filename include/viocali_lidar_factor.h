#include "parameters.h"
#include "utility/utility.h"
#include <ceres/ceres.h>
#include <condition_variable>
#include <map>
#include <math.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <stdio.h>
#include <thread>

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3>
SkewSymmetric(const Eigen::MatrixBase<Derived> &q) {
  Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
  ans << typename Derived::Scalar(0), -q(2), q(1), q(2),
      typename Derived::Scalar(0), -q(0), -q(1), q(0),
      typename Derived::Scalar(0);
  return ans;
}

template <typename T> static Eigen::Matrix<T, 3, 3> Exp(T v1, T v2, T v3) {
  T norm = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
  Eigen::Matrix<T, 3, 3> I;
  I << T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1);
  if (norm > 0.00001) {
    Eigen::Matrix<T, 3, 1> r_ang(v1 / norm, v2 / norm, v3 / norm);
    Eigen::Matrix<T, 3, 3> K = SkewSymmetric(r_ang);
    /// Roderigous Tranformation
    return I + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
  } else {
    return I;
  }
}

class LidarCostFunction : public ceres::SizedCostFunction<1, 3, 3, 3, 3> {
public:
  LidarCostFunction(const Eigen::Vector3d &world_point,
                    const Eigen::Vector3d &lidar_point,
                    const Eigen::Vector3d &plane_normal,
                    const double &plane_distance)
      : world_point_(world_point), lidar_point_(lidar_point),
        plane_normal_(plane_normal), plane_distance_(plane_distance){};

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d world_point_;
  Eigen::Vector3d lidar_point_;
  Eigen::Vector3d plane_normal_;
  double plane_distance_;
};

class RT : public ceres::SizedCostFunction<3, 3, 3> {
public:
  RT(const Eigen::Vector3d &world_point, const Eigen::Vector3d &lidar_point,
     const Eigen::Matrix3d &Rbl_, Eigen::Vector3d &tbl_)
      : world_point_(world_point), lidar_point_(lidar_point), Rbl(Rbl_),
        tbl(tbl_){};

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d world_point_;
  Eigen::Vector3d lidar_point_;
  Eigen::Matrix3d Rbl;
  Eigen::Vector3d tbl;
};

class EX : public ceres::SizedCostFunction<3, 3, 3> {
public:
  EX(const Eigen::Vector3d &world_point, const Eigen::Vector3d &lidar_point,
     const Eigen::Matrix3d &Rb0bk_, Eigen::Vector3d &tb0bk_)
      : world_point_(world_point), lidar_point_(lidar_point), Rb0bk(Rb0bk_),
        tb0bk(tb0bk_){};

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d world_point_;
  Eigen::Vector3d lidar_point_;
  Eigen::Matrix3d Rb0bk;
  Eigen::Vector3d tb0bk;
};

class LidarLocalParam : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override;

  virtual bool ComputeJacobian(const double *x,
                               double *jacobian) const override {
    ceres::MatrixRef(jacobian, 3, 3) = ceres::Matrix::Identity(3, 3);
    return true;
  }
  virtual int GlobalSize() const { return 3; };
  virtual int LocalSize() const { return 3; };
};

class LioCalib {
public:
  LioCalib(){};
  ~LioCalib(){};
  void Opitmize() { ceres::Problem p; };
};
