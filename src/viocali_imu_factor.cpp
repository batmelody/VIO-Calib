#include "viocali_imu_factor.h"

ImuIntegration::ImuIntegration(const Eigen::Vector3d &_acc_0,
                               const Eigen::Vector3d &_gyr_0,
                               const Eigen::Vector3d &_linearized_ba,
                               const Eigen::Vector3d &_linearized_bg)
    : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0},
      linearized_gyr{_gyr_0}, linearized_ba{_linearized_ba},
      linearized_bg{_linearized_bg},
      jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
      covariance{Eigen::Matrix<double, 15, 15>::Zero()}, sum_dt{0.0},
      delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()},
      delta_v{Eigen::Vector3d::Zero()} {
  noise = Eigen::Matrix<double, 18, 18>::Zero();
  noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
}

void ImuIntegration::push_back(double dt, const Eigen::Vector3d &acc,
                               const Eigen::Vector3d &gyr) {
  dt_buf.push_back(dt);
  acc_buf.push_back(acc);
  gyr_buf.push_back(gyr);
  propagate(dt, acc, gyr);
}

void ImuIntegration::repropagate(const Eigen::Vector3d &_linearized_ba,
                                 const Eigen::Vector3d &_linearized_bg) {
  sum_dt = 0.0;
  acc_0 = linearized_acc;
  gyr_0 = linearized_gyr;
  delta_p.setZero();
  delta_q.setIdentity();
  delta_v.setZero();
  linearized_ba = _linearized_ba;
  linearized_bg = _linearized_bg;
  jacobian.setIdentity();
  covariance.setZero();
  for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
    propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
}

void ImuIntegration::MidPointIntegration(
    double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
    const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
    const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
    const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
    const Eigen::Vector3d &linearized_bg, Eigen::Vector3d &result_delta_p,
    Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
    Eigen::Vector3d &result_linearized_ba,
    Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
  Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
  Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
  result_delta_q =
      delta_q * Eigen::Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2,
                                   un_gyr(2) * _dt / 2);
  Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
  result_delta_v = delta_v + un_acc * _dt;
  result_linearized_ba = linearized_ba;
  result_linearized_bg = linearized_bg;

  if (update_jacobian) {
    Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
    Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
    Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

    R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
    R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
        a_0_x(0), 0;
    R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
        a_1_x(0), 0;

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(0, 3) =
        -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
        -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
            (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
    F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(0, 9) =
        -0.25 *
        (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt *
        _dt;
    F.block<3, 3>(0, 12) =
        -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
    F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                          -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                              (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 9) =
        -0.5 *
        (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
    F.block<3, 3>(6, 12) =
        -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
    F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
    // cout<<"A"<<endl<<A<<endl;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
    V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
    V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x *
                          _dt * _dt * 0.5 * _dt;
    V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
    V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * _dt;
    V.block<3, 3>(3, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * _dt;
    V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
    V.block<3, 3>(6, 3) =
        0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
    V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
    V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3, 3) * _dt;
    V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3, 3) * _dt;

    // step_jacobian = F;
    // step_V = V;
    jacobian = F * jacobian;
    covariance = F * covariance * F.transpose() + V * noise * V.transpose();
  }
}

void ImuIntegration::propagate(double _dt, const Eigen::Vector3d &_acc_1,
                               const Eigen::Vector3d &_gyr_1) {
  dt = _dt;
  acc_1 = _acc_1;
  gyr_1 = _gyr_1;
  Eigen::Vector3d result_delta_p;
  Eigen::Quaterniond result_delta_q;
  Eigen::Vector3d result_delta_v;
  Eigen::Vector3d result_linearized_ba;
  Eigen::Vector3d result_linearized_bg;
  MidPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q,
                      delta_v, linearized_ba, linearized_bg, result_delta_p,
                      result_delta_q, result_delta_v, result_linearized_ba,
                      result_linearized_bg, 1);
  delta_p = result_delta_p;
  delta_q = result_delta_q;
  delta_v = result_delta_v;
  linearized_ba = result_linearized_ba;
  linearized_bg = result_linearized_bg;
  delta_q.normalize();
  sum_dt += dt;
  acc_0 = acc_1;
  gyr_0 = gyr_1;
}

Eigen::Matrix<double, 15, 1> ImuIntegration::evaluate(
    const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
    const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
    const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
    const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
    const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
  Eigen::Matrix<double, 15, 1> residuals;
  Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
  Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);
  Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
  Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
  Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);
  Eigen::Vector3d dba = Bai - linearized_ba;
  Eigen::Vector3d dbg = Bgi - linearized_bg;
  Eigen::Quaterniond corrected_delta_q =
      delta_q * Utility::deltaQ(dq_dbg * dbg);
  Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
  Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

  residuals.block<3, 1>(O_P, 0) =
      Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) -
      corrected_delta_p;
  residuals.block<3, 1>(O_R, 0) =
      2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
  residuals.block<3, 1>(O_V, 0) =
      Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
  residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
  residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
  return residuals;
}

bool ImuFactor::Evaluate(double const *const *parameters, double *residuals,
                         double **jacobians) const {

  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4],
                        parameters[0][5]);

  Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
  Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

  Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4],
                        parameters[2][5]);

  Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
  Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
  Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

  Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
  residual =
      PreIntegration->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

  Eigen::Matrix<double, 15, 15> sqrt_info =
      Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
          PreIntegration->covariance.inverse())
          .matrixL()
          .transpose();
  // sqrt_info.setIdentity();
  residual = sqrt_info * residual;

  if (jacobians) {
    double sum_dt = PreIntegration->sum_dt;
    Eigen::Matrix3d dp_dba =
        PreIntegration->jacobian.template block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg =
        PreIntegration->jacobian.template block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg =
        PreIntegration->jacobian.template block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba =
        PreIntegration->jacobian.template block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg =
        PreIntegration->jacobian.template block<3, 3>(O_V, O_BG);

    if (PreIntegration->jacobian.maxCoeff() > 1e8 ||
        PreIntegration->jacobian.minCoeff() < -1e8) {
    }

    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(
          jacobians[0]);
      jacobian_pose_i.setZero();

      jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
      jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(
          Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
      Eigen::Quaterniond corrected_delta_q =
          PreIntegration->delta_q *
          Utility::deltaQ(dq_dbg * (Bgi - PreIntegration->linearized_bg));
      jacobian_pose_i.block<3, 3>(O_R, O_R) =
          -(Utility::Qleft(Qj.inverse() * Qi) *
            Utility::Qright(corrected_delta_q))
               .bottomRightCorner<3, 3>();

      jacobian_pose_i.block<3, 3>(O_V, O_R) =
          Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

      jacobian_pose_i = sqrt_info * jacobian_pose_i;

      if (jacobian_pose_i.maxCoeff() > 1e8 ||
          jacobian_pose_i.minCoeff() < -1e8) {
      }
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>>
          jacobian_speedbias_i(jacobians[1]);
      jacobian_speedbias_i.setZero();
      jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) =
          -Qi.inverse().toRotationMatrix() * sum_dt;
      jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
      jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
      jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) =
          -Utility::Qleft(Qj.inverse() * Qi * PreIntegration->delta_q)
               .bottomRightCorner<3, 3>() *
          dq_dbg;

      jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) =
          -Qi.inverse().toRotationMatrix();
      jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
      jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

      jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) =
          -Eigen::Matrix3d::Identity();

      jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) =
          -Eigen::Matrix3d::Identity();

      jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(
          jacobians[2]);
      jacobian_pose_j.setZero();

      jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

      Eigen::Quaterniond corrected_delta_q =
          PreIntegration->delta_q *
          Utility::deltaQ(dq_dbg * (Bgi - PreIntegration->linearized_bg));
      jacobian_pose_j.block<3, 3>(O_R, O_R) =
          Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj)
              .bottomRightCorner<3, 3>();

      jacobian_pose_j = sqrt_info * jacobian_pose_j;
    }
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>>
          jacobian_speedbias_j(jacobians[3]);
      jacobian_speedbias_j.setZero();

      jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) =
          Qi.inverse().toRotationMatrix();

      jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) =
          Eigen::Matrix3d::Identity();

      jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) =
          Eigen::Matrix3d::Identity();

      jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
    }
  }
  return true;
}

ExRFactor::ExRFactor(const Eigen::Matrix3d &_R_c, const Eigen::Matrix3d &_R_b)
    : Rc(_R_c), Rb(_R_b){};

// /*useful process*/
// bool ExRFactor::Evaluate(double const *const *parameters, double *residuals,
//                          double **jacobians) const {
//   Eigen::Map<Eigen::Vector3d> residual(residuals);
//   Eigen::Vector3d p(1, 1, 1);
//   Sophus::Vector3d phi;
//   phi << parameters[0][0], parameters[0][1], parameters[0][2];
//   Sophus::SO3d R = Sophus::SO3d::exp(phi);
//   Eigen::Matrix3d Rbc = R.matrix();
//   residual = Rc * p - Rbc.transpose() * Rb * Rbc * p;
//   if (jacobians) {
//     if (jacobians[0]) {
//       Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_R(
//           jacobians[0]);
//       jacobian_R =
//           -1 * (-Rbc.transpose() * Rb * Rbc * Utility::skewSymmetric(p) +
//                 Utility::skewSymmetric(Rbc.transpose() * Rb * Rbc * p));
//     }
//   }
//   return true;
// }

bool ExRFactor::Evaluate(double const *const *parameters, double *residuals,
                         double **jacobians) const {
  Eigen::Map<Eigen::Vector3d> residual(residuals);
  Sophus::Vector3d phi;
  phi << parameters[0][0], parameters[0][1], parameters[0][2];
  Eigen::Quaterniond Qc(Rc);
  Qc = Qc.normalized();
  Eigen::Quaterniond Qb(Rb);
  Qb = Qb.normalized();
  Sophus::SO3d Rbc_SO3 = Sophus::SO3d::exp(phi);
  Sophus::SO3d Rc_SO3(Qc);
  Sophus::SO3d Rb_SO3(Qb);
  residual = (Rb_SO3.inverse() * Rbc_SO3 * Rc_SO3 * Rbc_SO3.inverse()).log();
  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_R(
          jacobians[0]);
      jacobian_R =
          -(Utility::Jright((Rbc_SO3 * Rc_SO3 * Rbc_SO3.inverse()).matrix()))
               .inverse() +
          Utility::Jleft((Rbc_SO3 * Rc_SO3 * Rbc_SO3.inverse()).matrix())
              .inverse();
    }
  }
  return true;
}
