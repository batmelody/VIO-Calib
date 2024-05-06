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