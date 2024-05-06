#pragma once
#include "include/utility/data_reader.h"
#include "viocali_lidar_factor.h"
#include <Eigen/Dense> // 包含Eigen库
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

void LidarDataSynthesis(std::deque<LidarData> &LidarData,
                        std::deque<Eigen::Vector3d> &world_points,
                        const double rb0bk[3], const double pb0bk[3]) {
  double parameters[4][3];
  parameters[0][0] = rb0bk[0];
  parameters[0][1] = rb0bk[1];
  parameters[0][2] = rb0bk[2];
  parameters[1][0] = pb0bk[0];
  parameters[1][1] = pb0bk[1];
  parameters[1][2] = pb0bk[2];
  parameters[2][0] = -1.112231;
  parameters[2][1] = -1.0899273;
  parameters[2][2] = 1.2874745;
  parameters[3][0] = -0.160473;
  parameters[3][1] = -0.043055002;
  parameters[3][2] = 0.040006001;
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

  for (int i = 0; i < LidarData.size(); i++) {
    Eigen::Vector3d lidar = LidarData[i].lidar_point;
    Eigen::Vector3d world = Rbl.transpose() * Rb0bk * (Rbl * lidar + tbl) +
                            Rbl.transpose() * tb0bk - Rbl.transpose() * tbl;
    world_points.push_back(world);
    LidarData[i].plane_distance = LidarData[i].plane_normal.transpose() * world;
  }
}

int main() {
  std::string LidarDir = "/home/zhengyuwei/Desktop/meas.txt";
  std::deque<LidarData> LidarGt = ReadLidarData(LidarDir);
  int FrameSize = 5;

  // init gt
  double parameters[4][3];
  parameters[0][0] = 0.0017697101;
  parameters[0][1] = -0.0017156728;
  parameters[0][2] = 0.70844859;
  parameters[1][0] = -0.26450777;
  parameters[1][1] = -0.22045621;
  parameters[1][2] = 0.0063481755;
  parameters[2][0] = -1.112231;
  parameters[2][1] = -1.0899273;
  parameters[2][2] = 1.2874745;
  parameters[3][0] = -0.160473;
  parameters[3][1] = -0.043055002;
  parameters[3][2] = 0.040006001;

  double initRb0bk[FrameSize][3];
  double initb0bk[FrameSize][3];
  double initRbl[3];
  double inittbl[3];

  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  loss_function = new ceres::CauchyLoss(1.0);
  LidarLocalParam *local_Pose_R = new LidarLocalParam();
  LidarLocalParam *local_Ex_R = new LidarLocalParam();

  for (int idx = 0; idx < FrameSize; idx++) {
    std::deque<Eigen::Vector3d> world_points;
    parameters[0][0] = 0.0017697101 + 0.05 * idx * pow(-1, idx);
    parameters[0][1] = -0.0017156728 - 0.05 * idx * pow(-1, idx);
    parameters[0][2] = 0.70844859 + 0.05 * idx * pow(-1, idx);
    parameters[1][0] = 0.0017697101 + 0.1 * idx * pow(-1, idx);
    parameters[1][1] = -0.0017156728 - 0.2 * idx * pow(-1, idx);
    parameters[1][2] = 0.70844859 + 0.1 * idx * pow(-1, idx);
    LidarDataSynthesis(LidarGt, world_points, parameters[0], parameters[1]);

    if (idx < 2) {
      // init Rb0bk
      initRb0bk[idx][0] = parameters[0][0];
      initRb0bk[idx][1] = parameters[0][1];
      initRb0bk[idx][2] = parameters[0][2];
      // init tb0bk
      initb0bk[idx][0] = parameters[1][0];
      initb0bk[idx][1] = parameters[1][1];
      initb0bk[idx][2] = parameters[1][2];
    } else {
      // init Rb0bk
      initRb0bk[idx][0] = parameters[0][0] - 0.2 * idx * pow(-1, idx);
      initRb0bk[idx][1] = parameters[0][1] - 0.1 * idx * pow(-1, idx);
      initRb0bk[idx][2] = parameters[0][2] + 0.1 * idx * pow(-1, idx);
      // init tb0bk
      initb0bk[idx][0] = parameters[1][0] - 0.2 * idx * pow(-1, idx);
      initb0bk[idx][1] = parameters[1][1] - 0.1 * idx * pow(-1, idx);
      initb0bk[idx][2] = parameters[1][2] + 0.3 * idx * pow(-1, idx);
    }

    std::cout << "before opt: " << parameters[0][0] << " " << parameters[0][1]
              << " " << parameters[0][2] << " " << parameters[1][0] << " "
              << parameters[1][1] << " " << parameters[1][2] << std::endl;

    // init exR
    initRbl[0] = parameters[2][0] + 0.1;
    initRbl[1] = parameters[2][1] - 0.1;
    initRbl[2] = parameters[2][2] + 0.2;

    // init exT
    inittbl[0] = parameters[3][0] + 0.3;
    inittbl[1] = parameters[3][1] - 0.5;
    inittbl[2] = parameters[3][2] - 0.1;

    problem.AddParameterBlock(initRbl, 3, local_Ex_R);
    problem.AddParameterBlock(initRb0bk[idx], 3, local_Pose_R);

    for (int i = 0; i < LidarGt.size(); i++) {
      ceres::CostFunction *costFunction = new LidarCostFunction(
          world_points[i], LidarGt[i].lidar_point, LidarGt[i].plane_normal,
          LidarGt[i].plane_distance);
      problem.AddResidualBlock(costFunction, loss_function, initRb0bk[idx],
                               initb0bk[idx], initRbl, inittbl);
    }
    if (idx == 0) {
      problem.SetParameterBlockConstant(initRb0bk[idx]);
      problem.SetParameterBlockConstant(initb0bk[idx]);
    }
    if (idx == 1) {
      problem.SetParameterBlockConstant(initb0bk[idx]);
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = true;
  options.use_nonmonotonic_steps = true;
  options.max_num_iterations = 100;
  options.function_tolerance = 1e-18;
  options.gradient_tolerance = 1e-18;
  options.parameter_tolerance = 1e-18;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  for (int idx = 0; idx < FrameSize; idx++) {
    std::cout << "after opt: " << initRb0bk[idx][0] << " " << initRb0bk[idx][1]
              << " " << initRb0bk[idx][2] << " " << initb0bk[idx][0] << " "
              << initb0bk[idx][1] << " " << initb0bk[idx][2] << std::endl;
  }
  std::cout << summary.BriefReport() << std::endl;
  std::cout << "Rlb " << initRbl[0] << ", " << initRbl[1] << ", " << initRbl[2]
            << std::endl;
  std::cout << "tlb " << inittbl[0] << ", " << inittbl[1] << ", " << inittbl[2]
            << std::endl;
  return 0;
}