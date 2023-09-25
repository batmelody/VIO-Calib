#pragma once

#include "utility/utility.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

class SophusLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override {
    const Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    const Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> x_plus_delta_lie(x_plus_delta);

    Sophus::SE3d T = Sophus::SE3d::exp(lie);
    Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);
    x_plus_delta_lie = (delta_T * T).log();
    return true;
  }

  virtual bool ComputeJacobian(const double *x,
                               double *jacobian) const override {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
  }

  virtual int GlobalSize() const { return Sophus::SE3d::DoF; }
  virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

class DistortionLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override {
    const Eigen::Map<const Eigen::Matrix<double, 2, 1>> K(x);
    const Eigen::Map<const Eigen::Matrix<double, 2, 1>> delta_K(delta);
    Eigen::Map<Eigen::Matrix<double, 2, 1>> x_plus_delta_K(x_plus_delta);

    x_plus_delta_K = K + delta_K;

    return true;
  }

  virtual bool ComputeJacobian(const double *x,
                               double *jacobian) const override {
    ceres::MatrixRef(jacobian, 2, 2) = ceres::Matrix::Identity(2, 2);
    return true;
  }
  virtual int GlobalSize() const { return 2; };
  virtual int LocalSize() const { return 2; };
};

class IntrinsicLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override {
    const Eigen::Map<const Eigen::Matrix<double, 4, 1>> K(x);
    const Eigen::Map<const Eigen::Matrix<double, 4, 1>> delta_K(delta);
    Eigen::Map<Eigen::Matrix<double, 4, 1>> x_plus_delta_K(x_plus_delta);

    x_plus_delta_K = K + delta_K;

    return true;
  }

  virtual bool ComputeJacobian(const double *x,
                               double *jacobian) const override {
    ceres::MatrixRef(jacobian, 4, 4) = ceres::Matrix::Identity(4, 4);
    return true;
  }
  virtual int GlobalSize() const { return 4; };
  virtual int LocalSize() const { return 4; };
};
