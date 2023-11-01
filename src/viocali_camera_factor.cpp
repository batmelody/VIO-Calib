#include "viocali_camera_factor.h"

ExtrinsicOnlyFactor::ExtrinsicOnlyFactor(const Eigen::Vector3d &_pts_i,
                                         const Eigen::Vector3d &_pts_j)
    : pts_i(_pts_i), pts_j(_pts_j){

                     };

// QuaternionFactor::QuaternionFactor(const Eigen::Quaterniond &Q_cam_,
//                                    const Eigen::Quaterniond &Q_imu_)
//     : Q_cam(Q_cam_), Q_imu(Q_imu_){

//                      };

EucmFactor::EucmFactor(const Eigen::Vector3d &_pts_i,
                       const Eigen::Vector3d &_pts_j)
    : pts_i(_pts_i), pts_j(_pts_j){

                     };

/*Sophus Version (WZY trust)*/
bool ExtrinsicOnlyFactor::Evaluate(double const *const *parameters,
                                   double *residuals,
                                   double **jacobians) const {

  Eigen::Map<Eigen::Vector3d> residual(residuals);

  Sophus::Vector6d Xi;
  Xi << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3],
      parameters[0][4], parameters[0][5];
  Sophus::SE3d T = Sophus::SE3d::exp(Xi);

  Eigen::Vector3d pts_j_pre = T * pts_i;
  residual = pts_j - pts_j_pre;
  double x = pts_j_pre[0];
  double y = pts_j_pre[1];
  double z = pts_j_pre[2];
  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian_pose_i(
          jacobians[0]);
      jacobian_pose_i.setZero();
      jacobian_pose_i(0, 0) = 1;
      jacobian_pose_i(0, 4) = z;
      jacobian_pose_i(0, 5) = -y;
      jacobian_pose_i(1, 1) = 1;
      jacobian_pose_i(1, 3) = -z;
      jacobian_pose_i(1, 5) = x;
      jacobian_pose_i(2, 2) = 1;
      jacobian_pose_i(2, 3) = y;
      jacobian_pose_i(2, 4) = -x;
      jacobian_pose_i = -jacobian_pose_i;
    }
  }
  return true;
}

/*Quaternion Valid (WZY trust)*/
// bool QuaternionFactor::Evaluate(double const *const *parameters,
//                                 double *residuals, double **jacobians) const
//                                 {

//   Eigen::Quaterniond Qic(parameters[0][0], parameters[0][1],
//   parameters[0][2],
//                          parameters[0][3]);

//   Eigen::Map<Eigen::Vector4d> residual(residuals);

//   return true;
// }

/*EUCM full-estimate*/
bool EucmFactor::Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const {

  Eigen::Map<Eigen::Vector2d> residual(residuals);

  Sophus::Vector6d Xi;
  Xi << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3],
      parameters[0][4], parameters[0][5];
  Sophus::SE3d T = Sophus::SE3d::exp(Xi);
  Eigen::Vector3d pts_c_pre = T * pts_i;
  double x = pts_c_pre[0];
  double y = pts_c_pre[1];
  double z = pts_c_pre[2];

  double alpha = parameters[1][0];
  double beta = parameters[1][1];

  double rho = sqrt(beta * (x * x + y * y) + z * z);
  double eta = alpha * rho + (1 - alpha) * z;

  Eigen::Vector3d pts_m_pre(x / (alpha * rho + (1 - alpha) * z),
                            y / (alpha * rho + (1 - alpha) * z), 1);

  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> dPm_dPc;
  dPm_dPc(0, 0) = 1 / eta - (alpha * beta * x * x) / ((eta * eta) * rho);
  dPm_dPc(0, 1) = (-alpha * beta * x * y) / ((eta * eta) * rho);
  dPm_dPc(0, 2) = -x / (eta * eta) * ((1 - alpha) + alpha * z / rho);
  dPm_dPc(1, 0) = (-alpha * beta * x * y) / ((eta * eta) * rho);
  dPm_dPc(1, 1) = 1 / eta - (alpha * beta * y * y) / ((eta * eta) * rho);
  dPm_dPc(1, 2) = -y / (eta * eta) * ((1 - alpha) + alpha * z / rho);

  Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dPc_dPw;
  dPc_dPw(0, 0) = 1;
  dPc_dPw(0, 4) = z;
  dPc_dPw(0, 5) = -y;
  dPc_dPw(1, 1) = 1;
  dPc_dPw(1, 3) = -z;
  dPc_dPw(1, 5) = x;
  dPc_dPw(2, 2) = 1;
  dPc_dPw(2, 3) = y;
  dPc_dPw(2, 4) = -x;

  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> dPm_dD;
  dPm_dD(0, 0) = -x / (eta * eta) * (rho - z);
  dPm_dD(0, 1) = -x / (eta * eta) * (alpha * (x * x + y * y) / (2 * rho));
  dPm_dD(1, 0) = -y / (eta * eta) * (rho - z);
  dPm_dD(1, 1) = -y / (eta * eta) * (alpha * (x * x + y * y) / (2 * rho));

  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> dp_dPm;
  dp_dPm(0, 0) = parameters[2][0];
  dp_dPm(0, 1) = 0;
  dp_dPm(1, 0) = 0;
  dp_dPm(1, 1) = parameters[2][1];

  Eigen::Matrix<double, 2, 4, Eigen::RowMajor> dp_dK;
  dp_dK(0, 0) = pts_m_pre.x();
  dp_dK(0, 1) = 0;
  dp_dK(0, 2) = 1;
  dp_dK(0, 3) = 0;
  dp_dK(1, 0) = 0;
  dp_dK(1, 1) = pts_m_pre.y();
  dp_dK(1, 2) = 0;
  dp_dK(1, 3) = 1;

  residual[0] = pts_j[0] - (pts_m_pre[0] * parameters[2][0] + parameters[2][2]);
  residual[1] = pts_j[1] - (pts_m_pre[1] * parameters[2][1] + parameters[2][3]);

  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(
          jacobians[0]);
      jacobian_pose_i.setZero();
      jacobian_pose_i = -1 * dp_dPm * dPm_dPc * dPc_dPw;
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> jacobian_dist_i(
          jacobians[1]);
      jacobian_dist_i.setZero();
      jacobian_dist_i = -1 * dp_dPm * dPm_dD;
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_dist_i(
          jacobians[2]);
      jacobian_dist_i.setZero();
      jacobian_dist_i = -1 * dp_dK;
    }
  }
  return true;
}

/*Camera Class*/
Camera::Parameters::Parameters(ModelType modelType)
    : m_modelType(modelType), m_imageWidth(0), m_imageHeight(0) {
  switch (modelType) {
  case FISHEYE:
    m_nIntrinsics = 8;
    break;
  case PINHOLE:
    m_nIntrinsics = 8;
    break;
  case MEI:
  default:
    m_nIntrinsics = 9;
  }
}

Camera::Parameters::Parameters(ModelType modelType,
                               const std::string &CameraName, int w, int h)
    : m_modelType(modelType), mCameraName(CameraName), m_imageWidth(w),
      m_imageHeight(h) {
  switch (modelType) {
  case FISHEYE:
    m_nIntrinsics = 8;
    break;
  case PINHOLE:
    m_nIntrinsics = 8;
    break;
  case MEI:
  default:
    m_nIntrinsics = 9;
  }
}

Camera::ModelType &Camera::Parameters::modelType(void) { return m_modelType; }

std::string &Camera::Parameters::CameraName(void) { return mCameraName; }

int &Camera::Parameters::imageWidth(void) { return m_imageWidth; }

int &Camera::Parameters::imageHeight(void) { return m_imageHeight; }

Camera::ModelType Camera::Parameters::modelType(void) const {
  return m_modelType;
}

const std::string &Camera::Parameters::CameraName(void) const {
  return mCameraName;
}

int Camera::Parameters::imageWidth(void) const { return m_imageWidth; }

int Camera::Parameters::imageHeight(void) const { return m_imageHeight; }

int Camera::Parameters::nIntrinsics(void) const { return m_nIntrinsics; }

cv::Mat &Camera::mask(void) { return m_mask; }

const cv::Mat &Camera::mask(void) const { return m_mask; }

void Camera::estimateExtrinsics(const std::vector<cv::Point3f> &objectPoints,
                                const std::vector<cv::Point2f> &imagePoints,
                                cv::Mat &rvec, cv::Mat &tvec) const {
  std::vector<cv::Point2f> Ms(imagePoints.size());
  for (size_t i = 0; i < Ms.size(); ++i) {
    Eigen::Vector3d P;
    liftProjective(Eigen::Vector2d(imagePoints.at(i).x, imagePoints.at(i).y),
                   P);

    P /= P(2);

    Ms.at(i).x = P(0);
    Ms.at(i).y = P(1);
  }

  // assume unit focal length, zero principal point, and zero distortion
  cv::solvePnP(objectPoints, Ms, cv::Mat::eye(3, 3, CV_64F), cv::noArray(),
               rvec, tvec);
}

double Camera::ReprojectionError(const Eigen::Vector3d &P1,
                                 const Eigen::Vector3d &P2) const {
  Eigen::Vector2d p1, p2;

  spaceToPlane(P1, p1);
  spaceToPlane(P2, p2);

  return (p1 - p2).norm();
}

double Camera::ReprojectionError(
    const std::vector<std::vector<cv::Point3f>> &objectPoints,
    const std::vector<std::vector<cv::Point2f>> &imagePoints,
    const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs,
    cv::OutputArray _perViewErrors) const {
  int imageCount = objectPoints.size();
  size_t pointsSoFar = 0;
  double totalErr = 0.0;

  bool computePerViewErrors = _perViewErrors.needed();
  cv::Mat perViewErrors;
  if (computePerViewErrors) {
    _perViewErrors.create(imageCount, 1, CV_64F);
    perViewErrors = _perViewErrors.getMat();
  }

  for (int i = 0; i < imageCount; ++i) {
    size_t pointCount = imagePoints.at(i).size();

    pointsSoFar += pointCount;

    std::vector<cv::Point2f> estImagePoints;
    projectPoints(objectPoints.at(i), rvecs.at(i), tvecs.at(i), estImagePoints);

    double err = 0.0;
    for (size_t j = 0; j < imagePoints.at(i).size(); ++j) {
      err += cv::norm(imagePoints.at(i).at(j) - estImagePoints.at(j));
    }

    if (computePerViewErrors) {
      perViewErrors.at<double>(i) = err / pointCount;
    }

    totalErr += err;
  }

  return totalErr / pointsSoFar;
}

double Camera::ReprojectionError(const Eigen::Vector3d &P,
                                 const Eigen::Quaterniond &camera_q,
                                 const Eigen::Vector3d &camera_t,
                                 const Eigen::Vector2d &observed_p) const {
  Eigen::Vector3d P_cam = camera_q.toRotationMatrix() * P + camera_t;

  Eigen::Vector2d p;
  spaceToPlane(P_cam, p);

  return (p - observed_p).norm();
}

void Camera::projectPoints(const std::vector<cv::Point3f> &objectPoints,
                           const cv::Mat &rvec, const cv::Mat &tvec,
                           std::vector<cv::Point2f> &imagePoints) const {
  // project 3D object points to the image plane
  imagePoints.reserve(objectPoints.size());

  // double
  cv::Mat R0;
  cv::Rodrigues(rvec, R0);

  Eigen::MatrixXd R(3, 3);
  R << R0.at<double>(0, 0), R0.at<double>(0, 1), R0.at<double>(0, 2),
      R0.at<double>(1, 0), R0.at<double>(1, 1), R0.at<double>(1, 2),
      R0.at<double>(2, 0), R0.at<double>(2, 1), R0.at<double>(2, 2);

  Eigen::Vector3d t;
  t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

  for (size_t i = 0; i < objectPoints.size(); ++i) {
    const cv::Point3f &objectPoint = objectPoints.at(i);

    // Rotate and translate
    Eigen::Vector3d P;
    P << objectPoint.x, objectPoint.y, objectPoint.z;

    P = R * P + t;

    Eigen::Vector2d p;
    spaceToPlane(P, p);

    imagePoints.push_back(cv::Point2f(p(0), p(1)));
  }
}

/* PinholeCamera*/
PinholeCamera::Parameters::Parameters()
    : Camera::Parameters(PINHOLE), m_k1(0.0), m_k2(0.0), m_p1(0.0), m_p2(0.0),
      m_fx(0.0), m_fy(0.0), m_cx(0.0), m_cy(0.0) {}

PinholeCamera::Parameters::Parameters(const std::string &CameraName, int w,
                                      int h, double k1, double k2, double p1,
                                      double p2, double fx, double fy,
                                      double cx, double cy)
    : Camera::Parameters(PINHOLE, CameraName, w, h), m_k1(k1), m_k2(k2),
      m_p1(p1), m_p2(p2), m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy) {}

double &PinholeCamera::Parameters::k1(void) { return m_k1; }

double &PinholeCamera::Parameters::k2(void) { return m_k2; }

double &PinholeCamera::Parameters::p1(void) { return m_p1; }

double &PinholeCamera::Parameters::p2(void) { return m_p2; }

double &PinholeCamera::Parameters::fx(void) { return m_fx; }

double &PinholeCamera::Parameters::fy(void) { return m_fy; }

double &PinholeCamera::Parameters::cx(void) { return m_cx; }

double &PinholeCamera::Parameters::cy(void) { return m_cy; }

double PinholeCamera::Parameters::k1(void) const { return m_k1; }

double PinholeCamera::Parameters::k2(void) const { return m_k2; }

double PinholeCamera::Parameters::p1(void) const { return m_p1; }

double PinholeCamera::Parameters::p2(void) const { return m_p2; }

double PinholeCamera::Parameters::fx(void) const { return m_fx; }

double PinholeCamera::Parameters::fy(void) const { return m_fy; }

double PinholeCamera::Parameters::cx(void) const { return m_cx; }

double PinholeCamera::Parameters::cy(void) const { return m_cy; }

bool PinholeCamera::Parameters::readFromYamlFile(const std::string &filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("PINHOLE") != 0) {
      return false;
    }
  }

  m_modelType = PINHOLE;
  fs["camera_name"] >> mCameraName;
  m_imageWidth = static_cast<int>(fs["image_width"]);
  m_imageHeight = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["distortion_parameters"];
  m_k1 = static_cast<double>(n["k1"]);
  m_k2 = static_cast<double>(n["k2"]);
  m_p1 = static_cast<double>(n["p1"]);
  m_p2 = static_cast<double>(n["p2"]);

  n = fs["projection_parameters"];
  m_fx = static_cast<double>(n["fx"]);
  m_fy = static_cast<double>(n["fy"]);
  m_cx = static_cast<double>(n["cx"]);
  m_cy = static_cast<double>(n["cy"]);

  return true;
}

void PinholeCamera::Parameters::writeToYamlFile(
    const std::string &filename) const {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "model_type"
     << "PINHOLE";
  fs << "camera_name" << mCameraName;
  fs << "image_width" << m_imageWidth;
  fs << "image_height" << m_imageHeight;

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  fs << "distortion_parameters";
  fs << "{"
     << "k1" << m_k1 << "k2" << m_k2 << "p1" << m_p1 << "p2" << m_p2 << "}";

  // projection: fx, fy, cx, cy
  fs << "projection_parameters";
  fs << "{"
     << "fx" << m_fx << "fy" << m_fy << "cx" << m_cx << "cy" << m_cy << "}";

  fs.release();
}

PinholeCamera::Parameters &
PinholeCamera::Parameters::operator=(const PinholeCamera::Parameters &other) {
  if (this != &other) {
    m_modelType = other.m_modelType;
    mCameraName = other.mCameraName;
    m_imageWidth = other.m_imageWidth;
    m_imageHeight = other.m_imageHeight;
    m_k1 = other.m_k1;
    m_k2 = other.m_k2;
    m_p1 = other.m_p1;
    m_p2 = other.m_p2;
    m_fx = other.m_fx;
    m_fy = other.m_fy;
    m_cx = other.m_cx;
    m_cy = other.m_cy;
  }

  return *this;
}

std::ostream &operator<<(std::ostream &out,
                         const PinholeCamera::Parameters &params) {
  out << "Camera Parameters:" << std::endl;
  out << "    model_type "
      << "PINHOLE" << std::endl;
  out << "   camera_name " << params.mCameraName << std::endl;
  out << "   image_width " << params.m_imageWidth << std::endl;
  out << "  image_height " << params.m_imageHeight << std::endl;

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  out << "Distortion Parameters" << std::endl;
  out << "            k1 " << params.m_k1 << std::endl
      << "            k2 " << params.m_k2 << std::endl
      << "            p1 " << params.m_p1 << std::endl
      << "            p2 " << params.m_p2 << std::endl;

  // projection: fx, fy, cx, cy
  out << "Projection Parameters" << std::endl;
  out << "            fx " << params.m_fx << std::endl
      << "            fy " << params.m_fy << std::endl
      << "            cx " << params.m_cx << std::endl
      << "            cy " << params.m_cy << std::endl;

  return out;
}

PinholeCamera::PinholeCamera()
    : m_inv_K11(1.0), m_inv_K13(0.0), m_inv_K22(1.0), m_inv_K23(0.0),
      m_noDistortion(true) {}

PinholeCamera::PinholeCamera(const std::string &CameraName, int imageWidth,
                             int imageHeight, double k1, double k2, double p1,
                             double p2, double fx, double fy, double cx,
                             double cy)
    : mParameters(CameraName, imageWidth, imageHeight, k1, k2, p1, p2, fx, fy,
                  cx, cy) {
  if ((mParameters.k1() == 0.0) && (mParameters.k2() == 0.0) &&
      (mParameters.p1() == 0.0) && (mParameters.p2() == 0.0)) {
    m_noDistortion = true;
  } else {
    m_noDistortion = false;
  }

  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.fx();
  m_inv_K13 = -mParameters.cx() / mParameters.fx();
  m_inv_K22 = 1.0 / mParameters.fy();
  m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

PinholeCamera::PinholeCamera(const PinholeCamera::Parameters &params)
    : mParameters(params) {
  if ((mParameters.k1() == 0.0) && (mParameters.k2() == 0.0) &&
      (mParameters.p1() == 0.0) && (mParameters.p2() == 0.0)) {
    m_noDistortion = true;
  } else {
    m_noDistortion = false;
  }

  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.fx();
  m_inv_K13 = -mParameters.cx() / mParameters.fx();
  m_inv_K22 = 1.0 / mParameters.fy();
  m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

Camera::ModelType PinholeCamera::modelType(void) const {
  return mParameters.modelType();
}

const std::string &PinholeCamera::CameraName(void) const {
  return mParameters.CameraName();
}

int PinholeCamera::imageWidth(void) const { return mParameters.imageWidth(); }

int PinholeCamera::imageHeight(void) const { return mParameters.imageHeight(); }

void PinholeCamera::estimateIntrinsics(
    const cv::Size &boardSize,
    const std::vector<std::vector<cv::Point3f>> &objectPoints,
    const std::vector<std::vector<cv::Point2f>> &imagePoints) {
  // Z. Zhang, A Flexible New Technique for Camera Calibration, PAMI 2000

  Parameters params = getParameters();

  params.k1() = 0.0;
  params.k2() = 0.0;
  params.p1() = 0.0;
  params.p2() = 0.0;

  double cx = params.imageWidth() / 2.0;
  double cy = params.imageHeight() / 2.0;
  params.cx() = cx;
  params.cy() = cy;

  size_t nImages = imagePoints.size();

  cv::Mat A(nImages * 2, 2, CV_64F);
  cv::Mat b(nImages * 2, 1, CV_64F);

  for (size_t i = 0; i < nImages; ++i) {
    const std::vector<cv::Point3f> &oPoints = objectPoints.at(i);

    std::vector<cv::Point2f> M(oPoints.size());
    for (size_t j = 0; j < M.size(); ++j) {
      M.at(j) = cv::Point2f(oPoints.at(j).x, oPoints.at(j).y);
    }

    cv::Mat H = cv::findHomography(M, imagePoints.at(i));

    H.at<double>(0, 0) -= H.at<double>(2, 0) * cx;
    H.at<double>(0, 1) -= H.at<double>(2, 1) * cx;
    H.at<double>(0, 2) -= H.at<double>(2, 2) * cx;
    H.at<double>(1, 0) -= H.at<double>(2, 0) * cy;
    H.at<double>(1, 1) -= H.at<double>(2, 1) * cy;
    H.at<double>(1, 2) -= H.at<double>(2, 2) * cy;

    double h[3], v[3], d1[3], d2[3];
    double n[4] = {0, 0, 0, 0};

    for (int j = 0; j < 3; ++j) {
      double t0 = H.at<double>(j, 0);
      double t1 = H.at<double>(j, 1);
      h[j] = t0;
      v[j] = t1;
      d1[j] = (t0 + t1) * 0.5;
      d2[j] = (t0 - t1) * 0.5;
      n[0] += t0 * t0;
      n[1] += t1 * t1;
      n[2] += d1[j] * d1[j];
      n[3] += d2[j] * d2[j];
    }

    for (int j = 0; j < 4; ++j) {
      n[j] = 1.0 / sqrt(n[j]);
    }

    for (int j = 0; j < 3; ++j) {
      h[j] *= n[0];
      v[j] *= n[1];
      d1[j] *= n[2];
      d2[j] *= n[3];
    }

    A.at<double>(i * 2, 0) = h[0] * v[0];
    A.at<double>(i * 2, 1) = h[1] * v[1];
    A.at<double>(i * 2 + 1, 0) = d1[0] * d2[0];
    A.at<double>(i * 2 + 1, 1) = d1[1] * d2[1];
    b.at<double>(i * 2, 0) = -h[2] * v[2];
    b.at<double>(i * 2 + 1, 0) = -d1[2] * d2[2];
  }

  cv::Mat f(2, 1, CV_64F);
  cv::solve(A, b, f, cv::DECOMP_NORMAL | cv::DECOMP_LU);

  params.fx() = sqrt(fabs(1.0 / f.at<double>(0)));
  params.fy() = sqrt(fabs(1.0 / f.at<double>(1)));

  setParameters(params);
}

/**
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void PinholeCamera::liftSphere(const Eigen::Vector2d &p,
                               Eigen::Vector3d &P) const {
  liftProjective(p, P);

  P.normalize();
}

/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void PinholeCamera::liftProjective(const Eigen::Vector2d &p,
                                   Eigen::Vector3d &P) const {
  double mx_d, my_d, mx2_d, mxy_d, my2_d, mx_u, my_u;
  double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
  // double lambda;

  // Lift points to normalised plane
  mx_d = m_inv_K11 * p(0) + m_inv_K13;
  my_d = m_inv_K22 * p(1) + m_inv_K23;

  if (m_noDistortion) {
    mx_u = mx_d;
    my_u = my_d;
  } else {
    if (0) {
      double k1 = mParameters.k1();
      double k2 = mParameters.k2();
      double p1 = mParameters.p1();
      double p2 = mParameters.p2();

      // Apply inverse distortion model
      // proposed by Heikkila
      mx2_d = mx_d * mx_d;
      my2_d = my_d * my_d;
      mxy_d = mx_d * my_d;
      rho2_d = mx2_d + my2_d;
      rho4_d = rho2_d * rho2_d;
      radDist_d = k1 * rho2_d + k2 * rho4_d;
      Dx_d = mx_d * radDist_d + p2 * (rho2_d + 2 * mx2_d) + 2 * p1 * mxy_d;
      Dy_d = my_d * radDist_d + p1 * (rho2_d + 2 * my2_d) + 2 * p2 * mxy_d;
      inv_denom_d = 1 / (1 + 4 * k1 * rho2_d + 6 * k2 * rho4_d + 8 * p1 * my_d +
                         8 * p2 * mx_d);

      mx_u = mx_d - inv_denom_d * Dx_d;
      my_u = my_d - inv_denom_d * Dy_d;
    } else {
      // Recursive distortion model
      int n = 8;
      Eigen::Vector2d d_u;
      distortion(Eigen::Vector2d(mx_d, my_d), d_u);
      // Approximate value
      mx_u = mx_d - d_u(0);
      my_u = my_d - d_u(1);

      for (int i = 1; i < n; ++i) {
        distortion(Eigen::Vector2d(mx_u, my_u), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
      }
    }
  }

  // Obtain a projective ray
  P << mx_u, my_u, 1.0;
}

/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void PinholeCamera::spaceToPlane(const Eigen::Vector3d &P,
                                 Eigen::Vector2d &p) const {
  Eigen::Vector2d p_u, p_d;

  // Project points to the normalised plane
  p_u << P(0) / P(2), P(1) / P(2);

  if (m_noDistortion) {
    p_d = p_u;
  } else {
    // Apply distortion
    Eigen::Vector2d d_u;
    distortion(p_u, d_u);
    p_d = p_u + d_u;
  }

  // Apply generalised projection matrix
  p << mParameters.fx() * p_d(0) + mParameters.cx(),
      mParameters.fy() * p_d(1) + mParameters.cy();
}

/**
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void PinholeCamera::undistToPlane(const Eigen::Vector2d &p_u,
                                  Eigen::Vector2d &p) const {
  Eigen::Vector2d p_d;

  if (m_noDistortion) {
    p_d = p_u;
  } else {
    // Apply distortion
    Eigen::Vector2d d_u;
    distortion(p_u, d_u);
    p_d = p_u + d_u;
  }

  // Apply generalised projection matrix
  p << mParameters.fx() * p_d(0) + mParameters.cx(),
      mParameters.fy() * p_d(1) + mParameters.cy();
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void PinholeCamera::distortion(const Eigen::Vector2d &p_u,
                               Eigen::Vector2d &d_u) const {
  double k1 = mParameters.k1();
  double k2 = mParameters.k2();
  double p1 = mParameters.p1();
  double p2 = mParameters.p2();

  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

  mx2_u = p_u(0) * p_u(0);
  my2_u = p_u(1) * p_u(1);
  mxy_u = p_u(0) * p_u(1);
  rho2_u = mx2_u + my2_u;
  rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
  d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
      p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *        and calculate Jacobian
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void PinholeCamera::distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u,
                               Eigen::Matrix2d &J) const {
  double k1 = mParameters.k1();
  double k2 = mParameters.k2();
  double p1 = mParameters.p1();
  double p2 = mParameters.p2();

  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

  mx2_u = p_u(0) * p_u(0);
  my2_u = p_u(1) * p_u(1);
  mxy_u = p_u(0) * p_u(1);
  rho2_u = mx2_u + my2_u;
  rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
  d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
      p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

  double dxdmx = 1.0 + rad_dist_u + k1 * 2.0 * mx2_u +
                 k2 * rho2_u * 4.0 * mx2_u + 2.0 * p1 * p_u(1) +
                 6.0 * p2 * p_u(0);
  double dydmx = k1 * 2.0 * p_u(0) * p_u(1) +
                 k2 * 4.0 * rho2_u * p_u(0) * p_u(1) + p1 * 2.0 * p_u(0) +
                 2.0 * p2 * p_u(1);
  double dxdmy = dydmx;
  double dydmy = 1.0 + rad_dist_u + k1 * 2.0 * my2_u +
                 k2 * rho2_u * 4.0 * my2_u + 6.0 * p1 * p_u(1) +
                 2.0 * p2 * p_u(0);

  J << dxdmx, dxdmy, dydmx, dydmy;
}

void PinholeCamera::initUndistortMap(cv::Mat &map1, cv::Mat &map2,
                                     double fScale) const {
  cv::Size imageSize(mParameters.imageWidth(), mParameters.imageHeight());

  cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
      double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

      Eigen::Vector3d P;
      P << mx_u, my_u, 1.0;

      Eigen::Vector2d p;
      spaceToPlane(P, p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}

cv::Mat PinholeCamera::initUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2,
                                               float fx, float fy,
                                               cv::Size imageSize, float cx,
                                               float cy, cv::Mat rmat) const {
  if (imageSize == cv::Size(0, 0)) {
    imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
  }

  cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

  Eigen::Matrix3f R, R_inv;
  cv::cv2eigen(rmat, R);
  R_inv = R.inverse();

  // assume no skew
  Eigen::Matrix3f K_rect;

  if (cx == -1.0f || cy == -1.0f) {
    K_rect << fx, 0, imageSize.width / 2, 0, fy, imageSize.height / 2, 0, 0, 1;
  } else {
    K_rect << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  }

  if (fx == -1.0f || fy == -1.0f) {
    K_rect(0, 0) = mParameters.fx();
    K_rect(1, 1) = mParameters.fy();
  }

  Eigen::Matrix3f K_rect_inv = K_rect.inverse();

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      Eigen::Vector3f xo;
      xo << u, v, 1;

      Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

      Eigen::Vector2d p;
      spaceToPlane(uo.cast<double>(), p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

  cv::Mat K_rect_cv;
  cv::eigen2cv(K_rect, K_rect_cv);
  return K_rect_cv;
}

int PinholeCamera::parameterCount(void) const { return 8; }

const PinholeCamera::Parameters &PinholeCamera::getParameters(void) const {
  return mParameters;
}

void PinholeCamera::setParameters(const PinholeCamera::Parameters &parameters) {
  mParameters = parameters;

  if ((mParameters.k1() == 0.0) && (mParameters.k2() == 0.0) &&
      (mParameters.p1() == 0.0) && (mParameters.p2() == 0.0)) {
    m_noDistortion = true;
  } else {
    m_noDistortion = false;
  }

  m_inv_K11 = 1.0 / mParameters.fx();
  m_inv_K13 = -mParameters.cx() / mParameters.fx();
  m_inv_K22 = 1.0 / mParameters.fy();
  m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

void PinholeCamera::readParameters(const std::vector<double> &parameterVec) {
  if ((int)parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.k1() = parameterVec.at(0);
  params.k2() = parameterVec.at(1);
  params.p1() = parameterVec.at(2);
  params.p2() = parameterVec.at(3);
  params.fx() = parameterVec.at(4);
  params.fy() = parameterVec.at(5);
  params.cx() = parameterVec.at(6);
  params.cy() = parameterVec.at(7);

  setParameters(params);
}

void PinholeCamera::writeParameters(std::vector<double> &parameterVec) const {
  parameterVec.resize(parameterCount());
  parameterVec.at(0) = mParameters.k1();
  parameterVec.at(1) = mParameters.k2();
  parameterVec.at(2) = mParameters.p1();
  parameterVec.at(3) = mParameters.p2();
  parameterVec.at(4) = mParameters.fx();
  parameterVec.at(5) = mParameters.fy();
  parameterVec.at(6) = mParameters.cx();
  parameterVec.at(7) = mParameters.cy();
}

void PinholeCamera::writeParametersToYamlFile(
    const std::string &filename) const {
  mParameters.writeToYamlFile(filename);
}

std::string PinholeCamera::parametersToString(void) const {
  std::ostringstream oss;
  oss << mParameters;

  return oss.str();
}

FisheyeCamera::Parameters::Parameters()
    : Camera::Parameters(FISHEYE), m_k2(0.0), m_k3(0.0), m_k4(0.0), m_k5(0.0),
      m_mu(0.0), m_mv(0.0), m_u0(0.0), m_v0(0.0) {}

FisheyeCamera::Parameters::Parameters(const std::string &CameraName, int w,
                                      int h, double k2, double k3, double k4,
                                      double k5, double mu, double mv,
                                      double u0, double v0)
    : Camera::Parameters(FISHEYE, CameraName, w, h), m_k2(k2), m_k3(k3),
      m_k4(k4), m_k5(k5), m_mu(mu), m_mv(mv), m_u0(u0), m_v0(v0) {}

double &FisheyeCamera::Parameters::k2(void) { return m_k2; }

double &FisheyeCamera::Parameters::k3(void) { return m_k3; }

double &FisheyeCamera::Parameters::k4(void) { return m_k4; }

double &FisheyeCamera::Parameters::k5(void) { return m_k5; }

double &FisheyeCamera::Parameters::mu(void) { return m_mu; }

double &FisheyeCamera::Parameters::mv(void) { return m_mv; }

double &FisheyeCamera::Parameters::u0(void) { return m_u0; }

double &FisheyeCamera::Parameters::v0(void) { return m_v0; }

double FisheyeCamera::Parameters::k2(void) const { return m_k2; }

double FisheyeCamera::Parameters::k3(void) const { return m_k3; }

double FisheyeCamera::Parameters::k4(void) const { return m_k4; }

double FisheyeCamera::Parameters::k5(void) const { return m_k5; }

double FisheyeCamera::Parameters::mu(void) const { return m_mu; }

double FisheyeCamera::Parameters::mv(void) const { return m_mv; }

double FisheyeCamera::Parameters::u0(void) const { return m_u0; }

double FisheyeCamera::Parameters::v0(void) const { return m_v0; }

bool FisheyeCamera::Parameters::readFromYamlFile(const std::string &filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("FISHEYE") != 0) {
      return false;
    }
  }

  m_modelType = FISHEYE;
  fs["camera_name"] >> mCameraName;
  m_imageWidth = static_cast<int>(fs["image_width"]);
  m_imageHeight = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["projection_parameters"];
  m_k2 = static_cast<double>(n["k2"]);
  m_k3 = static_cast<double>(n["k3"]);
  m_k4 = static_cast<double>(n["k4"]);
  m_k5 = static_cast<double>(n["k5"]);
  m_mu = static_cast<double>(n["mu"]);
  m_mv = static_cast<double>(n["mv"]);
  m_u0 = static_cast<double>(n["u0"]);
  m_v0 = static_cast<double>(n["v0"]);

  return true;
}

void FisheyeCamera::Parameters::writeToYamlFile(
    const std::string &filename) const {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "model_type"
     << "FISHEYE";
  fs << "camera_name" << mCameraName;
  fs << "image_width" << m_imageWidth;
  fs << "image_height" << m_imageHeight;

  // projection: k2, k3, k4, k5, mu, mv, u0, v0
  fs << "projection_parameters";
  fs << "{"
     << "k2" << m_k2 << "k3" << m_k3 << "k4" << m_k4 << "k5" << m_k5 << "mu"
     << m_mu << "mv" << m_mv << "u0" << m_u0 << "v0" << m_v0 << "}";

  fs.release();
}

FisheyeCamera::Parameters &
FisheyeCamera::Parameters::operator=(const FisheyeCamera::Parameters &other) {
  if (this != &other) {
    m_modelType = other.m_modelType;
    mCameraName = other.mCameraName;
    m_imageWidth = other.m_imageWidth;
    m_imageHeight = other.m_imageHeight;
    m_k2 = other.m_k2;
    m_k3 = other.m_k3;
    m_k4 = other.m_k4;
    m_k5 = other.m_k5;
    m_mu = other.m_mu;
    m_mv = other.m_mv;
    m_u0 = other.m_u0;
    m_v0 = other.m_v0;
  }

  return *this;
}

std::ostream &operator<<(std::ostream &out,
                         const FisheyeCamera::Parameters &params) {
  out << "Camera Parameters:" << std::endl;
  out << "    model_type "
      << "FISHEYE" << std::endl;
  out << "   camera_name " << params.mCameraName << std::endl;
  out << "   image_width " << params.m_imageWidth << std::endl;
  out << "  image_height " << params.m_imageHeight << std::endl;

  // projection: k2, k3, k4, k5, mu, mv, u0, v0
  out << "Projection Parameters" << std::endl;
  out << "            k2 " << params.m_k2 << std::endl
      << "            k3 " << params.m_k3 << std::endl
      << "            k4 " << params.m_k4 << std::endl
      << "            k5 " << params.m_k5 << std::endl
      << "            mu " << params.m_mu << std::endl
      << "            mv " << params.m_mv << std::endl
      << "            u0 " << params.m_u0 << std::endl
      << "            v0 " << params.m_v0 << std::endl;

  return out;
}

FisheyeCamera::FisheyeCamera()
    : m_inv_K11(1.0), m_inv_K13(0.0), m_inv_K22(1.0), m_inv_K23(0.0) {}

FisheyeCamera::FisheyeCamera(const std::string &CameraName, int imageWidth,
                             int imageHeight, double k2, double k3, double k4,
                             double k5, double mu, double mv, double u0,
                             double v0)
    : mParameters(CameraName, imageWidth, imageHeight, k2, k3, k4, k5, mu, mv,
                  u0, v0) {
  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.mu();
  m_inv_K13 = -mParameters.u0() / mParameters.mu();
  m_inv_K22 = 1.0 / mParameters.mv();
  m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

FisheyeCamera::FisheyeCamera(const FisheyeCamera::Parameters &params)
    : mParameters(params) {
  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.mu();
  m_inv_K13 = -mParameters.u0() / mParameters.mu();
  m_inv_K22 = 1.0 / mParameters.mv();
  m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

Camera::ModelType FisheyeCamera::modelType(void) const {
  return mParameters.modelType();
}

const std::string &FisheyeCamera::CameraName(void) const {
  return mParameters.CameraName();
}

int FisheyeCamera::imageWidth(void) const { return mParameters.imageWidth(); }

int FisheyeCamera::imageHeight(void) const { return mParameters.imageHeight(); }

void FisheyeCamera::estimateIntrinsics(
    const cv::Size &boardSize,
    const std::vector<std::vector<cv::Point3f>> &objectPoints,
    const std::vector<std::vector<cv::Point2f>> &imagePoints) {
  Parameters params = getParameters();

  double u0 = params.imageWidth() / 2.0;
  double v0 = params.imageHeight() / 2.0;

  double minReprojErr = std::numeric_limits<double>::max();

  std::vector<cv::Mat> rvecs, tvecs;
  rvecs.assign(objectPoints.size(), cv::Mat());
  tvecs.assign(objectPoints.size(), cv::Mat());

  params.k2() = 0.0;
  params.k3() = 0.0;
  params.k4() = 0.0;
  params.k5() = 0.0;
  params.u0() = u0;
  params.v0() = v0;

  // Initialize focal length
  // C. Hughes, P. Denny, M. Glavin, and E. Jones,
  // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
  // Extraction, PAMI 2010
  // Find circles from rows of chessboard corners, and for each pair
  // of circles, find vanishing points: v1 and v2.
  // f = ||v1 - v2|| / PI;
  // width:6 height:7
  double f0 = 0.0;
  std::cout << "boardSize: " << boardSize << std::endl;
  for (size_t i = 0; i < imagePoints.size(); ++i) {
    std::vector<Eigen::Vector2d> center(boardSize.height);
    double radius[boardSize.height];
    for (int r = 0; r < boardSize.height; ++r) {
      std::vector<cv::Point2d> circle; // image points of fisheye
      for (int c = 0; c < boardSize.width; ++c) {
        circle.push_back(imagePoints.at(i).at(r * boardSize.width +
                                              c)); // using ptr to write values
      }

      CalCircle(circle, center[r](0), center[r](1), radius[r]);
    }

    for (int j = 0; j < boardSize.height; ++j) {
      for (int k = j + 1; k < boardSize.height; ++k) {
        // find distance between pair of vanishing points which
        // correspond to intersection points of 2 circles
        std::vector<cv::Point2d> ipts;
        ipts = CircleIntersection(center[j](0), center[j](1), radius[j],
                                  center[k](0), center[k](1), radius[k]);

        if (ipts.size() < 2) {
          continue;
        }

        double f = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;

        params.mu() = f;
        params.mv() = f;

        setParameters(params);

        for (size_t l = 0; l < objectPoints.size(); ++l) {
          estimateExtrinsics(objectPoints.at(l), imagePoints.at(l), rvecs.at(l),
                             tvecs.at(l));
        }

        double reprojErr = ReprojectionError(objectPoints, imagePoints, rvecs,
                                             tvecs, cv::noArray());

        if (reprojErr < minReprojErr) {
          minReprojErr = reprojErr;
          f0 = f;
        }
      }
    }
  }

  if (f0 <= 0.0 && minReprojErr >= std::numeric_limits<double>::max()) {
    std::cout << "[" << params.CameraName() << "] "
              << "# INFO: kannala-Brandt model fails with given data. "
              << std::endl;

    return;
  }

  params.mu() = f0;
  params.mv() = f0;

  setParameters(params);
}

/**
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void FisheyeCamera::liftSphere(const Eigen::Vector2d &p,
                               Eigen::Vector3d &P) const {
  liftProjective(p, P);
}

/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void FisheyeCamera::liftProjective(const Eigen::Vector2d &p,
                                   Eigen::Vector3d &P) const {
  // Lift points to normalised plane
  Eigen::Vector2d p_u;
  p_u << m_inv_K11 * p(0) + m_inv_K13, m_inv_K22 * p(1) + m_inv_K23;

  // Obtain a projective ray
  double theta, phi;
  backprojectSymmetric(p_u, theta, phi);

  P(0) = sin(theta) * cos(phi);
  P(1) = sin(theta) * sin(phi);
  P(2) = cos(theta);
}

/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void FisheyeCamera::spaceToPlane(const Eigen::Vector3d &P,
                                 Eigen::Vector2d &p) const {
  double theta = acos(P(2) / P.norm());
  double phi = atan2(P(1), P(0));

  Eigen::Vector2d p_u = r(mParameters.k2(), mParameters.k3(), mParameters.k4(),
                          mParameters.k5(), theta) *
                        Eigen::Vector2d(cos(phi), sin(phi));

  // Apply generalised projection matrix
  p << mParameters.mu() * p_u(0) + mParameters.u0(),
      mParameters.mv() * p_u(1) + mParameters.v0();
}

/**
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void FisheyeCamera::spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p,
                                 Eigen::Matrix<double, 2, 3> &J) const {
  double theta = acos(P(2) / P.norm());
  double phi = atan2(P(1), P(0));

  Eigen::Vector2d p_u = r(mParameters.k2(), mParameters.k3(), mParameters.k4(),
                          mParameters.k5(), theta) *
                        Eigen::Vector2d(cos(phi), sin(phi));

  // Apply generalised projection matrix
  p << mParameters.mu() * p_u(0) + mParameters.u0(),
      mParameters.mv() * p_u(1) + mParameters.v0();
}

/**
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void FisheyeCamera::undistToPlane(const Eigen::Vector2d &p_u,
                                  Eigen::Vector2d &p) const {
  //    Eigen::Vector2d p_d;
  //
  //    if (m_noDistortion)
  //    {
  //        p_d = p_u;
  //    }
  //    else
  //    {
  //        // Apply distortion
  //        Eigen::Vector2d d_u;
  //        distortion(p_u, d_u);
  //        p_d = p_u + d_u;
  //    }
  //
  //    // Apply generalised projection matrix
  //    p << mParameters.gamma1() * p_d(0) + mParameters.u0(),
  //         mParameters.gamma2() * p_d(1) + mParameters.v0();
}

void FisheyeCamera::initUndistortMap(cv::Mat &map1, cv::Mat &map2,
                                     double fScale) const {
  cv::Size imageSize(mParameters.imageWidth(), mParameters.imageHeight());

  cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
      double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

      double theta, phi;
      backprojectSymmetric(Eigen::Vector2d(mx_u, my_u), theta, phi);

      Eigen::Vector3d P;
      P << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);

      Eigen::Vector2d p;
      spaceToPlane(P, p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}

cv::Mat FisheyeCamera::initUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2,
                                               float fx, float fy,
                                               cv::Size imageSize, float cx,
                                               float cy, cv::Mat rmat) const {
  if (imageSize == cv::Size(0, 0)) {
    imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
  }

  cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

  Eigen::Matrix3f K_rect;

  if (cx == -1.0f && cy == -1.0f) {
    K_rect << fx, 0, imageSize.width / 2, 0, fy, imageSize.height / 2, 0, 0, 1;
  } else {
    K_rect << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  }

  if (fx == -1.0f || fy == -1.0f) {
    K_rect(0, 0) = mParameters.mu();
    K_rect(1, 1) = mParameters.mv();
  }

  Eigen::Matrix3f K_rect_inv = K_rect.inverse();

  Eigen::Matrix3f R, R_inv;
  cv::cv2eigen(rmat, R);
  R_inv = R.inverse();

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      Eigen::Vector3f xo;
      xo << u, v, 1;

      Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

      Eigen::Vector2d p;
      spaceToPlane(uo.cast<double>(), p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

  cv::Mat K_rect_cv;
  cv::eigen2cv(K_rect, K_rect_cv);
  return K_rect_cv;
}

int FisheyeCamera::parameterCount(void) const { return 8; }

const FisheyeCamera::Parameters &FisheyeCamera::getParameters(void) const {
  return mParameters;
}

void FisheyeCamera::setParameters(const FisheyeCamera::Parameters &parameters) {
  mParameters = parameters;

  // Inverse camera projection matrix parameters
  m_inv_K11 = 1.0 / mParameters.mu();
  m_inv_K13 = -mParameters.u0() / mParameters.mu();
  m_inv_K22 = 1.0 / mParameters.mv();
  m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

void FisheyeCamera::readParameters(const std::vector<double> &parameterVec) {
  if (parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.k2() = parameterVec.at(0);
  params.k3() = parameterVec.at(1);
  params.k4() = parameterVec.at(2);
  params.k5() = parameterVec.at(3);
  params.mu() = parameterVec.at(4);
  params.mv() = parameterVec.at(5);
  params.u0() = parameterVec.at(6);
  params.v0() = parameterVec.at(7);
  std::cout << "new Intrin: " << parameterVec.at(4) << ", "
            << parameterVec.at(5) << ", " << parameterVec.at(6) << ", "
            << parameterVec.at(7);
  setParameters(params);
}

void FisheyeCamera::writeParameters(std::vector<double> &parameterVec) const {
  parameterVec.resize(parameterCount());
  parameterVec.at(0) = mParameters.k2();
  parameterVec.at(1) = mParameters.k3();
  parameterVec.at(2) = mParameters.k4();
  parameterVec.at(3) = mParameters.k5();
  parameterVec.at(4) = mParameters.mu();
  parameterVec.at(5) = mParameters.mv();
  parameterVec.at(6) = mParameters.u0();
  parameterVec.at(7) = mParameters.v0();
}

void FisheyeCamera::writeParametersToYamlFile(
    const std::string &filename) const {
  mParameters.writeToYamlFile(filename);
}

std::string FisheyeCamera::parametersToString(void) const {
  std::ostringstream oss;
  oss << mParameters;

  return oss.str();
}

void FisheyeCamera::fitOddPoly(const std::vector<double> &x,
                               const std::vector<double> &y, int n,
                               std::vector<double> &coeffs) const {
  std::vector<int> pows;
  for (int i = 1; i <= n; i += 2) {
    pows.push_back(i);
  }

  Eigen::MatrixXd X(x.size(), pows.size());
  Eigen::MatrixXd Y(y.size(), 1);
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < pows.size(); ++j) {
      X(i, j) = pow(x.at(i), pows.at(j));
    }
    Y(i, 0) = y.at(i);
  }

  Eigen::MatrixXd A = (X.transpose() * X).inverse() * X.transpose() * Y;

  coeffs.resize(A.rows());
  for (int i = 0; i < A.rows(); ++i) {
    coeffs.at(i) = A(i, 0);
  }
}

std::vector<cv::Point2d> FisheyeCamera::CircleIntersection(double x1, double y1,
                                                           double r1, double x2,
                                                           double y2,
                                                           double r2) {
  std::vector<cv::Point2d> ipts;

  double d = hypot(x1 - x2, y1 - y2);
  if (d > r1 + r2) {
    // circles are separate
    return ipts;
  }
  if (d < fabs(r1 - r2)) {
    // one circle is contained within the other
    return ipts;
  }

  double a = (square(r1) - square(r2) + square(d)) / (2.0 * d);
  double h = sqrt(square(r1) - square(a));

  double x3 = x1 + a * (x2 - x1) / d;
  double y3 = y1 + a * (y2 - y1) / d;

  if (h < 1e-10) {
    // two circles touch at one point
    ipts.push_back(cv::Point2d(x3, y3));
    return ipts;
  }

  ipts.push_back(cv::Point2d(x3 + h * (y2 - y1) / d, y3 - h * (x2 - x1) / d));
  ipts.push_back(cv::Point2d(x3 - h * (y2 - y1) / d, y3 + h * (x2 - x1) / d));
  return ipts;
}

void FisheyeCamera::CalCircle(const std::vector<cv::Point2d> &points,
                              double &centerX, double &centerY,
                              double &radius) {
  // D. Umbach, and K. Jones, A Few Methods for Fitting Circles to Data,
  // IEEE Transactions on Instrumentation and Measurement, 2000
  // We use the modified least squares method.
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xx = 0.0;
  double sum_xy = 0.0;
  double sum_yy = 0.0;
  double sum_xxx = 0.0;
  double sum_xxy = 0.0;
  double sum_xyy = 0.0;
  double sum_yyy = 0.0;

  int n = points.size();
  for (int i = 0; i < n; ++i) {
    double x = points.at(i).x;
    double y = points.at(i).y;

    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_xy += x * y;
    sum_yy += y * y;
    sum_xxx += x * x * x;
    sum_xxy += x * x * y;
    sum_xyy += x * y * y;
    sum_yyy += y * y * y;
  }

  double A = n * sum_xx - square(sum_x);
  double B = n * sum_xy - sum_x * sum_y;
  double C = n * sum_yy - square(sum_y);
  double D =
      0.5 * (n * sum_xyy - sum_x * sum_yy + n * sum_xxx - sum_x * sum_xx);
  double E =
      0.5 * (n * sum_xxy - sum_y * sum_xx + n * sum_yyy - sum_y * sum_yy);

  centerX = (D * C - B * E) / (A * C - square(B));
  centerY = (A * E - B * D) / (A * C - square(B));

  double sum_r = 0.0;
  for (int i = 0; i < n; ++i) {
    double x = points.at(i).x;
    double y = points.at(i).y;

    sum_r += hypot(x - centerX, y - centerY);
  }

  radius = sum_r / n;
}

void FisheyeCamera::backprojectSymmetric(const Eigen::Vector2d &p_u,
                                         double &theta, double &phi) const {
  double tol = 1e-10;
  double p_u_norm = p_u.norm();

  if (p_u_norm < 1e-10) {
    phi = 0.0;
  } else {
    phi = atan2(p_u(1), p_u(0));
  }

  int npow = 9;
  if (mParameters.k5() == 0.0) {
    npow -= 2;
  }
  if (mParameters.k4() == 0.0) {
    npow -= 2;
  }
  if (mParameters.k3() == 0.0) {
    npow -= 2;
  }
  if (mParameters.k2() == 0.0) {
    npow -= 2;
  }

  Eigen::MatrixXd coeffs(npow + 1, 1);
  coeffs.setZero();
  coeffs(0) = -p_u_norm;
  coeffs(1) = 1.0;

  if (npow >= 3) {
    coeffs(3) = mParameters.k2();
  }
  if (npow >= 5) {
    coeffs(5) = mParameters.k3();
  }
  if (npow >= 7) {
    coeffs(7) = mParameters.k4();
  }
  if (npow >= 9) {
    coeffs(9) = mParameters.k5();
  }

  if (npow == 1) {
    theta = p_u_norm;
  } else {
    // Get eigenvalues of companion matrix corresponding to polynomial.
    // Eigenvalues correspond to roots of polynomial.
    Eigen::MatrixXd A(npow, npow);
    A.setZero();
    A.block(1, 0, npow - 1, npow - 1).setIdentity();
    A.col(npow - 1) = -coeffs.block(0, 0, npow, 1) / coeffs(npow);

    Eigen::EigenSolver<Eigen::MatrixXd> es(A);
    Eigen::MatrixXcd eigval = es.eigenvalues();

    std::vector<double> thetas;
    for (int i = 0; i < eigval.rows(); ++i) {
      if (fabs(eigval(i).imag()) > tol) {
        continue;
      }

      double t = eigval(i).real();

      if (t < -tol) {
        continue;
      } else if (t < 0.0) {
        t = 0.0;
      }

      thetas.push_back(t);
    }

    if (thetas.empty()) {
      theta = p_u_norm;
    } else {
      theta = *std::min_element(thetas.begin(), thetas.end());
    }
  }
}

/* CostFunction */
boost::shared_ptr<CostFunction> CostFunction::CostFunctionInstance;
boost::shared_ptr<CostFunction> CostFunction::instance(void) {
  if (CostFunctionInstance.get() == 0) {
    CostFunctionInstance.reset(new CostFunction);
  }

  return CostFunctionInstance;
}
template <typename T>
void worldToCameraTransform(const T *const q_cam_odo, const T *const t_cam_odo,
                            const T *const p_odo, const T *const att_odo, T *q,
                            T *t, bool optimize_cam_odo_z = true) {
  Eigen::Quaternion<T> q_z_inv(cos(att_odo[0] / T(2)), T(0), T(0),
                               -sin(att_odo[0] / T(2)));
  Eigen::Quaternion<T> q_y_inv(cos(att_odo[1] / T(2)), T(0),
                               -sin(att_odo[1] / T(2)), T(0));
  Eigen::Quaternion<T> q_x_inv(cos(att_odo[2] / T(2)), -sin(att_odo[2] / T(2)),
                               T(0), T(0));

  Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;

  T q_odo[4] = {q_zyx_inv.w(), q_zyx_inv.x(), q_zyx_inv.y(), q_zyx_inv.z()};

  T q_odo_cam[4] = {q_cam_odo[3], -q_cam_odo[0], -q_cam_odo[1], -q_cam_odo[2]};

  T q0[4];
  ceres::QuaternionProduct(q_odo_cam, q_odo, q0);

  T t0[3];
  T t_odo[3] = {p_odo[0], p_odo[1], p_odo[2]};

  ceres::QuaternionRotatePoint(q_odo, t_odo, t0);

  t0[0] += t_cam_odo[0];
  t0[1] += t_cam_odo[1];

  if (optimize_cam_odo_z) {
    t0[2] += t_cam_odo[2];
  }

  ceres::QuaternionRotatePoint(q_odo_cam, t0, t);
  t[0] = -t[0];
  t[1] = -t[1];
  t[2] = -t[2];

  // Convert quaternion from Ceres convention (w, x, y, z)
  // to Eigen convention (x, y, z, w)
  q[0] = q0[1];
  q[1] = q0[2];
  q[2] = q0[3];
  q[3] = q0[0];
}

template <class CameraT> class ReprojectionError1 {
public:
  ReprojectionError1(const Eigen::Vector3d &observed_P,
                     const Eigen::Vector2d &observed_p)
      : m_observed_P(observed_P), m_observed_p(observed_p),
        m_sqrtPrecisionMat(Eigen::Matrix2d::Identity()) {}

  ReprojectionError1(const Eigen::Vector3d &observed_P,
                     const Eigen::Vector2d &observed_p,
                     const Eigen::Matrix2d &sqrtPrecisionMat)
      : m_observed_P(observed_P), m_observed_p(observed_p),
        m_sqrtPrecisionMat(sqrtPrecisionMat) {}

  ReprojectionError1(const std::vector<double> &intrinsic_params,
                     const Eigen::Vector3d &observed_P,
                     const Eigen::Vector2d &observed_p)
      : m_intrinsic_params(intrinsic_params), m_observed_P(observed_P),
        m_observed_p(observed_p) {}

  // variables: camera intrinsics and camera extrinsics
  template <typename T>
  bool operator()(const T *const intrinsic_params, const T *const q,
                  const T *const t, T *residuals) const {
    Eigen::Matrix<T, 3, 1> P = m_observed_P.cast<T>();

    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params, q, t, P, predicted_p);

    Eigen::Matrix<T, 2, 1> e = predicted_p - m_observed_p.cast<T>();

    Eigen::Matrix<T, 2, 1> e_weighted = m_sqrtPrecisionMat.cast<T>() * e;

    residuals[0] = e_weighted(0);
    residuals[1] = e_weighted(1);

    return true;
  }

  // variables: camera-odometry transforms and odometry poses
  template <typename T>
  bool operator()(const T *const q_cam_odo, const T *const t_cam_odo,
                  const T *const p_odo, const T *const att_odo,
                  T *residuals) const {
    T q[4], t[3];
    worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t);

    Eigen::Matrix<T, 3, 1> P = m_observed_P.cast<T>();

    std::vector<T> intrinsic_params(m_intrinsic_params.begin(),
                                    m_intrinsic_params.end());

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

    residuals[0] = predicted_p(0) - T(m_observed_p(0));
    residuals[1] = predicted_p(1) - T(m_observed_p(1));

    return true;
  }

  // private:
  //  camera intrinsics
  std::vector<double> m_intrinsic_params;

  // observed 3D point
  Eigen::Vector3d m_observed_P;

  // observed 2D point
  Eigen::Vector2d m_observed_p;

  // square root of precision matrix
  Eigen::Matrix2d m_sqrtPrecisionMat;
};

template <class CameraTypeLeft, class CameraTypeRight>
class StereoReprojectionError {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoReprojectionError(const Eigen::Vector3d &observed_P,
                          const Eigen::Vector2d &observed_p_l,
                          const Eigen::Vector2d &observed_p_r)
      : m_observed_P(observed_P), m_observed_p_l(observed_p_l),
        m_observed_p_r(observed_p_r) {}

  template <typename T>
  bool operator()(const T *const intrinsic_params_l,
                  const T *const intrinsic_params_r, const T *const q_l,
                  const T *const t_l, const T *const q_l_r,
                  const T *const t_l_r, T *residuals) const {
    Eigen::Matrix<T, 3, 1> P;
    P(0) = T(m_observed_P(0));
    P(1) = T(m_observed_P(1));
    P(2) = T(m_observed_P(2));

    Eigen::Matrix<T, 2, 1> predicted_p_l;
    CameraTypeLeft::spaceToPlane(intrinsic_params_l, q_l, t_l, P,
                                 predicted_p_l);

    Eigen::Quaternion<T> q_r =
        Eigen::Quaternion<T>(q_l_r) * Eigen::Quaternion<T>(q_l);

    Eigen::Matrix<T, 3, 1> t_r;
    t_r(0) = t_l[0];
    t_r(1) = t_l[1];
    t_r(2) = t_l[2];

    t_r = Eigen::Quaternion<T>(q_l_r) * t_r;
    t_r(0) += t_l_r[0];
    t_r(1) += t_l_r[1];
    t_r(2) += t_l_r[2];

    Eigen::Matrix<T, 2, 1> predicted_p_r;
    CameraTypeRight::spaceToPlane(intrinsic_params_r, q_r.coeffs().data(),
                                  t_r.data(), P, predicted_p_r);

    residuals[0] = predicted_p_l(0) - T(m_observed_p_l(0));
    residuals[1] = predicted_p_l(1) - T(m_observed_p_l(1));
    residuals[2] = predicted_p_r(0) - T(m_observed_p_r(0));
    residuals[3] = predicted_p_r(1) - T(m_observed_p_r(1));

    return true;
  }

private:
  // observed 3D point
  Eigen::Vector3d m_observed_P;

  // observed 2D point
  Eigen::Vector2d m_observed_p_l;
  Eigen::Vector2d m_observed_p_r;
};

ceres::CostFunction *
CostFunction::CreateCostFunction(const CameraConstPtr &camera,
                                 const Eigen::Vector3d &observed_P,
                                 const Eigen::Vector2d &observed_p) const {
  ceres::CostFunction *costFunction = 0;
  std::vector<double> intrinsic_params;
  switch (camera->modelType()) {
  case Camera::FISHEYE:
    costFunction =
        new ceres::AutoDiffCostFunction<ReprojectionError1<FisheyeCamera>, 2, 8,
                                        4, 3>(
            new ReprojectionError1<FisheyeCamera>(observed_P, observed_p));
    break;
  case Camera::PINHOLE:
    costFunction =
        new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8,
                                        4, 3>(
            new ReprojectionError1<PinholeCamera>(observed_P, observed_p));
    break;
  }
  return costFunction;
}

ceres::CostFunction *CostFunction::CreateCostFunction(
    const CameraConstPtr &cameraL, const CameraConstPtr &cameraR,
    const Eigen::Vector3d &observed_P, const Eigen::Vector2d &observed_p_l,
    const Eigen::Vector2d &observed_p_r) const {
  ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<
      StereoReprojectionError<PinholeCamera, FisheyeCamera>, 4, 8, 8, 4, 3, 4,
      3>(new StereoReprojectionError<PinholeCamera, FisheyeCamera>(
      observed_P, observed_p_l, observed_p_r));

  return costFunction;
}

CameraCalibration::CameraCalibration()
    : mBoardSize(cv::Size(0, 0)), mSquareSize(0.0f), mVerBose(false) {}

CameraCalibration::CameraCalibration(Camera::ModelType modelType,
                                     const std::string &CameraName,
                                     const cv::Size &imageSize,
                                     const cv::Size &boardSize,
                                     float squareSize)
    : mBoardSize(boardSize), mSquareSize(squareSize), mVerBose(false) {
  mCamera =
      GeneralCamera::instance()->CreateCamera(modelType, CameraName, imageSize);
}

CameraCalibration::CameraCalibration(
    Camera::ModelType modelTypeLeft, Camera::ModelType modelTypeRight,
    const std::string &CameraNameLeft, const std::string &CameraNameRight,
    const cv::Size &imageSizeLeft, const cv::Size &imageSizeRight,
    const cv::Size &boardSize, float squareSize)
    : mBoardSize(boardSize), mSquareSize(squareSize), mVerBose(false) {
  mCamera_left = GeneralCamera::instance()->CreateCamera(
      modelTypeLeft, CameraNameLeft, imageSizeLeft);
  mCamera_right = GeneralCamera::instance()->CreateCamera(
      modelTypeRight, CameraNameRight, imageSizeRight);
}

void CameraCalibration::clear(void) {
  mImagePoints.clear();
  mScenePoints.clear();
}

bool CameraCalibration::findChessboardCorners(
    const cv::Mat &image, const cv::Size &patternSize,
    std::vector<cv::Point2f> &corners) {
  return cv::findChessboardCorners(
      image, patternSize, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
          cv::CALIB_CB_FILTER_QUADS + cv::CALIB_CB_FAST_CHECK);
}

void CameraCalibration::addChessboardData(
    std::vector<std::vector<cv::Point3f>> &world_pts,
    std::vector<std::vector<cv::Point2f>> &img_pts) {
  mScenePoints = world_pts;
  mImagePoints = img_pts;
}

void CameraCalibration::addStereoChessboardData(
    std::vector<std::vector<cv::Point3f>> &world_pts,
    std::vector<std::vector<cv::Point2f>> &img_pts_left,
    std::vector<std::vector<cv::Point2f>> &img_pts_right) {
  mScenePoints = world_pts;
  mImagePoints_left = img_pts_left;
  mImagePoints_right = img_pts_right;
}

bool CameraCalibration::calibrate(void) {
  int imageCount = mImagePoints.size();
  std::vector<std::vector<cv::Point2f>> image_points = mImagePoints;

  // compute intrinsic camera parameters and extrinsic parameters for each of
  // the views
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  bool ret = CalibrationExecute(mCamera, image_points, rvecs, tvecs);

  mCameraPoses = cv::Mat(imageCount, 6, CV_64F);
  for (int i = 0; i < imageCount; ++i) {
    mCameraPoses.at<double>(i, 0) = rvecs.at(i).at<double>(0);
    mCameraPoses.at<double>(i, 1) = rvecs.at(i).at<double>(1);
    mCameraPoses.at<double>(i, 2) = rvecs.at(i).at<double>(2);
    mCameraPoses.at<double>(i, 3) = tvecs.at(i).at<double>(0);
    mCameraPoses.at<double>(i, 4) = tvecs.at(i).at<double>(1);
    mCameraPoses.at<double>(i, 5) = tvecs.at(i).at<double>(2);
  }

  // Compute measurement covariance.
  std::vector<std::vector<cv::Point2f>> errVec(mImagePoints.size());
  Eigen::Vector2d errSum = Eigen::Vector2d::Zero();
  size_t errCount = 0;
  for (size_t i = 0; i < mImagePoints.size(); ++i) {
    std::vector<cv::Point2f> estImagePoints;
    mCamera->projectPoints(mScenePoints.at(i), rvecs.at(i), tvecs.at(i),
                           estImagePoints);

    for (size_t j = 0; j < mImagePoints.at(i).size(); ++j) {
      cv::Point2f pObs = mImagePoints.at(i).at(j);
      cv::Point2f pEst = estImagePoints.at(j);

      cv::Point2f err = pObs - pEst;

      errVec.at(i).push_back(err);

      errSum += Eigen::Vector2d(err.x, err.y);
    }
    errCount += mImagePoints.at(i).size();
  }

  Eigen::Vector2d errMean = errSum / static_cast<double>(errCount);

  Eigen::Matrix2d measurementCovariance = Eigen::Matrix2d::Zero();
  for (size_t i = 0; i < errVec.size(); ++i) {
    for (size_t j = 0; j < errVec.at(i).size(); ++j) {
      cv::Point2f err = errVec.at(i).at(j);
      double d0 = err.x - errMean(0);
      double d1 = err.y - errMean(1);

      measurementCovariance(0, 0) += d0 * d0;
      measurementCovariance(0, 1) += d0 * d1;
      measurementCovariance(1, 1) += d1 * d1;
    }
  }
  measurementCovariance /= static_cast<double>(errCount);
  measurementCovariance(1, 0) = measurementCovariance(0, 1);

  mMeasurementCovariance = measurementCovariance;

  return ret;
}

bool CameraCalibration::calibrateStereo(void) {
  int imageCount = mImagePoints_left.size();
  // compute intrinsic camera parameters and extrinsic parameters for each of
  // the views
  std::vector<cv::Mat> rvecs_left;
  std::vector<cv::Mat> tvecs_left;
  std::vector<cv::Mat> rvecs_right;
  std::vector<cv::Mat> tvecs_right;
  std::vector<std::vector<cv::Point2f>> image_points_left = mImagePoints_left;
  std::vector<std::vector<cv::Point2f>> image_points_right = mImagePoints_right;
  bool ret = StereoCalibrationExecute(
      mCamera_left, mCamera_right, image_points_left, image_points_right,
      rvecs_left, tvecs_left, rvecs_right, tvecs_right);
  return ret;
}

int CameraCalibration::sampleCount(void) const { return mImagePoints.size(); }

std::vector<std::vector<cv::Point2f>> &CameraCalibration::imagePoints(void) {
  return mImagePoints;
}

const std::vector<std::vector<cv::Point2f>> &
CameraCalibration::imagePoints(void) const {
  return mImagePoints;
}

std::vector<std::vector<cv::Point3f>> &CameraCalibration::scenePoints(void) {
  return mScenePoints;
}

const std::vector<std::vector<cv::Point3f>> &
CameraCalibration::scenePoints(void) const {
  return mScenePoints;
}

CameraPtr &CameraCalibration::camera(void) { return mCamera; }

const CameraConstPtr CameraCalibration::camera(void) const { return mCamera; }

Eigen::Matrix2d &CameraCalibration::measurementCovariance(void) {
  return mMeasurementCovariance;
}

const Eigen::Matrix2d &CameraCalibration::measurementCovariance(void) const {
  return mMeasurementCovariance;
}

cv::Mat &CameraCalibration::cameraPoses(void) { return mCameraPoses; }

const cv::Mat &CameraCalibration::cameraPoses(void) const {
  return mCameraPoses;
}

void CameraCalibration::drawResults(std::vector<cv::Mat> &images) const {
  std::vector<cv::Mat> rvecs, tvecs;

  for (size_t i = 0; i < images.size(); ++i) {
    cv::Mat rvec(3, 1, CV_64F);
    rvec.at<double>(0) = mCameraPoses.at<double>(i, 0);
    rvec.at<double>(1) = mCameraPoses.at<double>(i, 1);
    rvec.at<double>(2) = mCameraPoses.at<double>(i, 2);

    cv::Mat tvec(3, 1, CV_64F);
    tvec.at<double>(0) = mCameraPoses.at<double>(i, 3);
    tvec.at<double>(1) = mCameraPoses.at<double>(i, 4);
    tvec.at<double>(2) = mCameraPoses.at<double>(i, 5);

    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
  }

  int drawShiftBits = 4;
  int drawMultiplier = 1 << drawShiftBits;

  cv::Scalar green(0, 255, 0);
  cv::Scalar red(0, 0, 255);

  for (size_t i = 0; i < images.size(); ++i) {
    cv::Mat &image = images.at(i);
    if (image.channels() == 1) {
      cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }

    std::vector<cv::Point2f> estImagePoints;
    mCamera->projectPoints(mScenePoints.at(i), rvecs.at(i), tvecs.at(i),
                           estImagePoints);

    float errorSum = 0.0f;
    float errorMax = std::numeric_limits<float>::min();

    for (size_t j = 0; j < mImagePoints.at(i).size(); ++j) {
      cv::Point2f pObs = mImagePoints.at(i).at(j);
      cv::Point2f pEst = estImagePoints.at(j);

      cv::circle(image,
                 cv::Point(cvRound(pObs.x * drawMultiplier),
                           cvRound(pObs.y * drawMultiplier)),
                 5, green, 2, cv::LINE_AA, drawShiftBits);

      cv::circle(image,
                 cv::Point(cvRound(pEst.x * drawMultiplier),
                           cvRound(pEst.y * drawMultiplier)),
                 5, red, 2, cv::LINE_AA, drawShiftBits);

      float error = cv::norm(pObs - pEst);

      errorSum += error;
      if (error > errorMax) {
        errorMax = error;
      }
    }

    std::ostringstream oss;
    oss << "Reprojection error: avg = " << errorSum / mImagePoints.at(i).size()
        << "   max = " << errorMax;

    cv::putText(image, oss.str(), cv::Point(10, image.rows - 10),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);
  }
}

void CameraCalibration::writeParams(const std::string &filename) const {
  mCamera->writeParametersToYamlFile(filename);
}

bool CameraCalibration::writeChessboardData(const std::string &filename) const {
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    return false;
  }

  WriteData(ofs, mBoardSize.width);
  WriteData(ofs, mBoardSize.height);
  WriteData(ofs, mSquareSize);

  WriteData(ofs, mMeasurementCovariance(0, 0));
  WriteData(ofs, mMeasurementCovariance(0, 1));
  WriteData(ofs, mMeasurementCovariance(1, 0));
  WriteData(ofs, mMeasurementCovariance(1, 1));

  WriteData(ofs, mCameraPoses.rows);
  WriteData(ofs, mCameraPoses.cols);
  WriteData(ofs, mCameraPoses.type());
  for (int i = 0; i < mCameraPoses.rows; ++i) {
    for (int j = 0; j < mCameraPoses.cols; ++j) {
      WriteData(ofs, mCameraPoses.at<double>(i, j));
    }
  }

  WriteData(ofs, mImagePoints.size());
  for (size_t i = 0; i < mImagePoints.size(); ++i) {
    WriteData(ofs, mImagePoints.at(i).size());
    for (size_t j = 0; j < mImagePoints.at(i).size(); ++j) {
      const cv::Point2f &ipt = mImagePoints.at(i).at(j);

      WriteData(ofs, ipt.x);
      WriteData(ofs, ipt.y);
    }
  }

  WriteData(ofs, mScenePoints.size());
  for (size_t i = 0; i < mScenePoints.size(); ++i) {
    WriteData(ofs, mScenePoints.at(i).size());
    for (size_t j = 0; j < mScenePoints.at(i).size(); ++j) {
      const cv::Point3f &spt = mScenePoints.at(i).at(j);

      WriteData(ofs, spt.x);
      WriteData(ofs, spt.y);
      WriteData(ofs, spt.z);
    }
  }

  return true;
}

bool CameraCalibration::readChessboardData(const std::string &filename) {
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }

  ReadData(ifs, mBoardSize.width);
  ReadData(ifs, mBoardSize.height);
  ReadData(ifs, mSquareSize);

  ReadData(ifs, mMeasurementCovariance(0, 0));
  ReadData(ifs, mMeasurementCovariance(0, 1));
  ReadData(ifs, mMeasurementCovariance(1, 0));
  ReadData(ifs, mMeasurementCovariance(1, 1));

  int rows, cols, type;
  ReadData(ifs, rows);
  ReadData(ifs, cols);
  ReadData(ifs, type);
  mCameraPoses = cv::Mat(rows, cols, type);

  for (int i = 0; i < mCameraPoses.rows; ++i) {
    for (int j = 0; j < mCameraPoses.cols; ++j) {
      ReadData(ifs, mCameraPoses.at<double>(i, j));
    }
  }

  size_t nImagePointSets;
  ReadData(ifs, nImagePointSets);

  mImagePoints.clear();
  mImagePoints.resize(nImagePointSets);
  for (size_t i = 0; i < mImagePoints.size(); ++i) {
    size_t nImagePoints;
    ReadData(ifs, nImagePoints);
    mImagePoints.at(i).resize(nImagePoints);

    for (size_t j = 0; j < mImagePoints.at(i).size(); ++j) {
      cv::Point2f &ipt = mImagePoints.at(i).at(j);
      ReadData(ifs, ipt.x);
      ReadData(ifs, ipt.y);
    }
  }

  size_t nScenePointSets;
  ReadData(ifs, nScenePointSets);

  mScenePoints.clear();
  mScenePoints.resize(nScenePointSets);
  for (size_t i = 0; i < mScenePoints.size(); ++i) {
    size_t nScenePoints;
    ReadData(ifs, nScenePoints);
    mScenePoints.at(i).resize(nScenePoints);

    for (size_t j = 0; j < mScenePoints.at(i).size(); ++j) {
      cv::Point3f &spt = mScenePoints.at(i).at(j);
      ReadData(ifs, spt.x);
      ReadData(ifs, spt.y);
      ReadData(ifs, spt.z);
    }
  }

  return true;
}

void CameraCalibration::setVerbose(bool verbose) { mVerBose = verbose; }

bool CameraCalibration::CalibrationExecute(
    CameraPtr &camera, std::vector<std::vector<cv::Point2f>> &image_points,
    std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs) const {
  rvecs.assign(image_points.size(), cv::Mat());
  tvecs.assign(image_points.size(), cv::Mat());

  // STEP 1: Estimate intrinsics
  camera->estimateIntrinsics(mBoardSize, mScenePoints, image_points);

  // STEP 2: Estimate extrinsics
  for (size_t i = 0; i < mScenePoints.size(); ++i) {
    camera->estimateExtrinsics(mScenePoints.at(i), image_points.at(i),
                               rvecs.at(i), tvecs.at(i));
  }

  if (mCamera) {
    std::cout << "[" << camera->CameraName() << "] "
              << "# INFO: "
              << "Initial reprojection error: " << std::fixed
              << std::setprecision(3)
              << camera->ReprojectionError(mScenePoints, image_points, rvecs,
                                           tvecs)
              << " pixels" << std::endl;
  }

  // STEP 3: optimization using ceres
  MonoOptimizer(camera, image_points, rvecs, tvecs);

  if (mCamera) {
    double err =
        camera->ReprojectionError(mScenePoints, image_points, rvecs, tvecs);
    std::cout << "[" << camera->CameraName() << "] "
              << "# INFO: Final reprojection error: " << err << " pixels"
              << std::endl;
    std::cout << "[" << camera->CameraName() << "] "
              << "# INFO: " << camera->parametersToString() << std::endl;
  }
  return true;
}

bool CameraCalibration::StereoCalibrationExecute(
    CameraPtr &camera_left, CameraPtr &camera_right,
    std::vector<std::vector<cv::Point2f>> &image_points_left,
    std::vector<std::vector<cv::Point2f>> &image_points_right,
    std::vector<cv::Mat> &rvecs_left, std::vector<cv::Mat> &tvecs_left,
    std::vector<cv::Mat> &rvecs_right,
    std::vector<cv::Mat> &tvecs_right) const {
  rvecs_left.assign(mScenePoints.size(), cv::Mat());
  tvecs_left.assign(mScenePoints.size(), cv::Mat());
  rvecs_right.assign(mScenePoints.size(), cv::Mat());
  tvecs_right.assign(mScenePoints.size(), cv::Mat());

  // STEP 1: Estimate Left intrinsics
  camera_left->estimateIntrinsics(mBoardSize, mScenePoints, image_points_left);

  // STEP 2: Estimate Left extrinsics
  for (size_t i = 0; i < mScenePoints.size(); ++i) {
    camera_left->estimateExtrinsics(mScenePoints.at(i), image_points_left.at(i),
                                    rvecs_left.at(i), tvecs_left.at(i));
  }
  MonoOptimizer(camera_left, image_points_left, rvecs_left, tvecs_left);

  // STEP 3: Estimate Right intrinsics
  camera_right->estimateIntrinsics(mBoardSize, mScenePoints,
                                   image_points_right);

  // STEP 4: Estimate Right extrinsics
  for (size_t i = 0; i < mScenePoints.size(); ++i) {
    camera_right->estimateExtrinsics(mScenePoints.at(i),
                                     image_points_right.at(i),
                                     rvecs_right.at(i), tvecs_right.at(i));
  }
  MonoOptimizer(camera_right, image_points_right, rvecs_right, tvecs_right);

  StereoOptimizer(camera_left, camera_right, image_points_left,
                  image_points_right, rvecs_left, tvecs_left, rvecs_right,
                  tvecs_right);

  double err = camera_left->ReprojectionError(mScenePoints, image_points_left,
                                              rvecs_left, tvecs_left);
  std::cout << "[" << camera_left->CameraName() << "] "
            << "# INFO: Final reprojection error: " << err << " pixels"
            << std::endl;
  std::cout << "[" << camera_left->CameraName() << "] "
            << "# INFO: " << camera_left->parametersToString() << std::endl;

  double errr = camera_right->ReprojectionError(
      mScenePoints, image_points_right, rvecs_right, tvecs_right);
  std::cout << "[" << camera_right->CameraName() << "] "
            << "# INFO: Final reprojection error: " << errr << " pixels"
            << std::endl;
  std::cout << "[" << camera_right->CameraName() << "] "
            << "# INFO: " << camera_right->parametersToString() << std::endl;
  return true;
}

void CameraCalibration::MonoOptimizer(
    CameraPtr &camera, std::vector<std::vector<cv::Point2f>> &image_points,
    std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs) const {
  // Use ceres to do optimization
  ceres::Problem problem;
  std::vector<Transform, Eigen::aligned_allocator<Transform>> transformVec(
      rvecs.size());
  for (size_t i = 0; i < rvecs.size(); ++i) {
    Eigen::Vector3d rvec;
    cv::cv2eigen(rvecs.at(i), rvec);

    transformVec.at(i).rotation() =
        Eigen::AngleAxisd(rvec.norm(), rvec.normalized());
    transformVec.at(i).translation() << tvecs[i].at<double>(0),
        tvecs[i].at<double>(1), tvecs[i].at<double>(2);
  }

  std::vector<double> intrinsicCameraParams;
  camera->writeParameters(intrinsicCameraParams);

  // create residuals for each observation
  for (size_t i = 0; i < image_points.size(); ++i) {
    for (size_t j = 0; j < image_points.at(i).size(); ++j) {
      const cv::Point3f &spt = mScenePoints.at(i).at(j);
      const cv::Point2f &ipt = image_points.at(i).at(j);

      ceres::CostFunction *costFunction =
          CostFunction::instance()->CreateCostFunction(
              camera, Eigen::Vector3d(spt.x, spt.y, spt.z),
              Eigen::Vector2d(ipt.x, ipt.y));

      ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
      problem.AddResidualBlock(costFunction, lossFunction,
                               intrinsicCameraParams.data(),
                               transformVec.at(i).rotationData(),
                               transformVec.at(i).translationData());
    }

    ceres::LocalParameterization *quaternionParameterization =
        new EigenQuaternionParameterization;

    problem.SetParameterization(transformVec.at(i).rotationData(),
                                quaternionParameterization);
  }

  std::cout << "begin ceres" << std::endl;
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.num_threads = 1;

  if (mVerBose) {
    options.minimizer_progress_to_stdout = true;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << "end ceres" << std::endl;

  if (mVerBose) {
    std::cout << summary.FullReport() << std::endl;
  }

  camera->readParameters(intrinsicCameraParams);

  for (size_t i = 0; i < rvecs.size(); ++i) {
    Eigen::AngleAxisd aa(transformVec.at(i).rotation());

    Eigen::Vector3d rvec = aa.angle() * aa.axis();
    cv::eigen2cv(rvec, rvecs.at(i));

    cv::Mat &tvec = tvecs.at(i);
    tvec.at<double>(0) = transformVec.at(i).translation()(0);
    tvec.at<double>(1) = transformVec.at(i).translation()(1);
    tvec.at<double>(2) = transformVec.at(i).translation()(2);
  }
}

void CameraCalibration::StereoOptimizer(
    CameraPtr &camera_left, CameraPtr &camera_right,
    std::vector<std::vector<cv::Point2f>> &image_points_left,
    std::vector<std::vector<cv::Point2f>> &image_points_right,
    std::vector<cv::Mat> &rvecs_left, std::vector<cv::Mat> &tvecs_left,
    std::vector<cv::Mat> &rvecs_right,
    std::vector<cv::Mat> &tvecs_right) const {
  ceres::Problem problem;

  // Process Left RT
  std::vector<Transform, Eigen::aligned_allocator<Transform>> transformVecLeft(
      rvecs_left.size());
  for (size_t i = 0; i < rvecs_left.size(); ++i) {
    Eigen::Vector3d rvec;
    cv::cv2eigen(rvecs_left.at(i), rvec);
    transformVecLeft.at(i).rotation() =
        Eigen::AngleAxisd(rvec.norm(), rvec.normalized());
    transformVecLeft.at(i).translation() << tvecs_left[i].at<double>(0),
        tvecs_left[i].at<double>(1), tvecs_left[i].at<double>(2);
  }

  // Process Left to the Right RT
  std::vector<Transform, Eigen::aligned_allocator<Transform>> Extrinsic(1);
  Eigen::Vector3d rvec;
  rvec << rvecs_left[0].at<double>(0, 0), rvecs_left[0].at<double>(0, 1),
      rvecs_left[0].at<double>(0, 2);
  Eigen::Quaterniond q_l = Extrinsic.at(0).AngleAxisToQuaternion(rvec);
  Eigen::Vector3d t_l;
  t_l << tvecs_left[0].at<double>(0, 0), tvecs_left[0].at<double>(0, 1),
      tvecs_left[0].at<double>(0, 2);
  rvec << rvecs_right[0].at<double>(0, 0), rvecs_right[0].at<double>(0, 1),
      rvecs_right[0].at<double>(0, 2);
  Eigen::Quaterniond q_r = Extrinsic.at(0).AngleAxisToQuaternion(rvec);
  Eigen::Vector3d t_r;
  t_r << tvecs_right[0].at<double>(0, 0), tvecs_right[0].at<double>(0, 1),
      tvecs_right[0].at<double>(0, 2);
  Eigen::Quaterniond q_l_r = q_r * q_l.conjugate();
  Eigen::Vector3d t_l_r = -q_l_r.toRotationMatrix() * t_l + t_r;

  Extrinsic.at(0).rotation() = q_l_r;
  Extrinsic.at(0).translation() = t_l_r;

  std::cout << " init t_left_right" << t_l_r << std::endl;

  std::vector<double> intrinsicCameraParamsLeft;
  std::vector<double> intrinsicCameraParamsRight;
  mCamera_left->writeParameters(intrinsicCameraParamsLeft);
  mCamera_right->writeParameters(intrinsicCameraParamsRight);

  // create residuals for each observation
  for (size_t i = 0; i < image_points_left.size(); ++i) {
    for (size_t j = 0; j < image_points_left.at(i).size(); ++j) {
      const cv::Point3f &spt = mScenePoints.at(i).at(j);
      const cv::Point2f &iptl = image_points_left.at(i).at(j);
      const cv::Point2f &iptr = image_points_right.at(i).at(j);
      ceres::CostFunction *costFunction =
          CostFunction::instance()->CreateCostFunction(
              camera_left, camera_right, Eigen::Vector3d(spt.x, spt.y, spt.z),
              Eigen::Vector2d(iptl.x, iptl.y), Eigen::Vector2d(iptr.x, iptr.y));
      ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
      problem.AddResidualBlock(
          costFunction, lossFunction, intrinsicCameraParamsLeft.data(),
          intrinsicCameraParamsRight.data(),
          transformVecLeft.at(i).rotationData(),
          transformVecLeft.at(i).translationData(),
          Extrinsic.at(0).rotationData(), Extrinsic.at(0).translationData());
    }
  }

  for (int i = 0; i < image_points_left.size(); ++i) {
    ceres::LocalParameterization *quaternionParameterization =
        new EigenQuaternionParameterization;

    problem.SetParameterization(transformVecLeft.at(i).rotationData(),
                                quaternionParameterization);
  }

  ceres::LocalParameterization *quaternionParameterization =
      new EigenQuaternionParameterization;

  problem.SetParameterization(Extrinsic.at(0).rotationData(),
                              quaternionParameterization);
  std::cout << "begin ceres" << std::endl;
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.num_threads = 1;

  if (mVerBose) {
    options.minimizer_progress_to_stdout = true;
  }

  camera_left->readParameters(intrinsicCameraParamsLeft);
  camera_right->readParameters(intrinsicCameraParamsRight);
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << "end ceres" << std::endl;

  std::cout << "Extrinsic.at(i).translation(): "
            << Extrinsic.at(0).translation() << std::endl;
}

template <typename T>
void CameraCalibration::ReadData(std::ifstream &ifs, T &data) const {
  char *buffer = new char[sizeof(T)];

  ifs.read(buffer, sizeof(T));

  data = *(reinterpret_cast<T *>(buffer));

  delete[] buffer;
}

template <typename T>
void CameraCalibration::WriteData(std::ofstream &ofs, T data) const {
  char *pData = reinterpret_cast<char *>(&data);

  ofs.write(pData, sizeof(T));
}

/*Transform class*/
Transform::Transform() {
  m_q.setIdentity();
  m_t.setZero();
}

Transform::Transform(const Eigen::Matrix4d &H) {
  m_q = Eigen::Quaterniond(H.block<3, 3>(0, 0));
  m_t = H.block<3, 1>(0, 3);
}

Eigen::Quaterniond &Transform::rotation(void) { return m_q; }

const Eigen::Quaterniond &Transform::rotation(void) const { return m_q; }

double *Transform::rotationData(void) { return m_q.coeffs().data(); }

const double *const Transform::rotationData(void) const {
  return m_q.coeffs().data();
}

Eigen::Vector3d &Transform::translation(void) { return m_t; }

const Eigen::Vector3d &Transform::translation(void) const { return m_t; }

double *Transform::translationData(void) { return m_t.data(); }

const double *const Transform::translationData(void) const {
  return m_t.data();
}

Eigen::Matrix4d Transform::toMatrix(void) const {
  Eigen::Matrix4d H;
  H.setIdentity();
  H.block<3, 3>(0, 0) = m_q.toRotationMatrix();
  H.block<3, 1>(0, 3) = m_t;

  return H;
}

template <typename T>
Eigen::Matrix<T, 3, 3>
Transform::AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1> &rvec) {
  T angle = rvec.norm();
  if (angle == T(0)) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }

  Eigen::Matrix<T, 3, 1> axis;
  axis = rvec.normalized();

  Eigen::Matrix<T, 3, 3> rmat;
  rmat = Eigen::AngleAxis<T>(angle, axis);

  return rmat;
}

template <typename T>
Eigen::Quaternion<T>
Transform::AngleAxisToQuaternion(const Eigen::Matrix<T, 3, 1> &rvec) {
  Eigen::Matrix<T, 3, 3> rmat = AngleAxisToRotationMatrix<T>(rvec);

  return Eigen::Quaternion<T>(rmat);
}

/*LocalParameter class*/
bool EigenQuaternionParameterization::Plus(const double *x, const double *delta,
                                           double *x_plus_delta) const {
  const double norm_delta =
      sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  if (norm_delta > 0.0) {
    const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
    double q_delta[4];
    q_delta[0] = sin_delta_by_delta * delta[0];
    q_delta[1] = sin_delta_by_delta * delta[1];
    q_delta[2] = sin_delta_by_delta * delta[2];
    q_delta[3] = cos(norm_delta);
    EigenQuaternionProduct(q_delta, x, x_plus_delta);
  } else {
    for (int i = 0; i < 4; ++i) {
      x_plus_delta[i] = x[i];
    }
  }
  return true;
}

bool EigenQuaternionParameterization::ComputeJacobian(const double *x,
                                                      double *jacobian) const {
  jacobian[0] = x[3];
  jacobian[1] = x[2];
  jacobian[2] = -x[1]; // NOLINT
  jacobian[3] = -x[2];
  jacobian[4] = x[3];
  jacobian[5] = x[0]; // NOLINT
  jacobian[6] = x[1];
  jacobian[7] = -x[0];
  jacobian[8] = x[3]; // NOLINT
  jacobian[9] = -x[0];
  jacobian[10] = -x[1];
  jacobian[11] = -x[2]; // NOLINT
  return true;
}

/*GeneralCamera*/
boost::shared_ptr<GeneralCamera> GeneralCamera::GeneralCameraInstanceinstance;
boost::shared_ptr<GeneralCamera> GeneralCamera::instance(void) {
  if (GeneralCameraInstanceinstance.get() == 0) {
    GeneralCameraInstanceinstance.reset(new GeneralCamera);
  }
  return GeneralCameraInstanceinstance;
}

GeneralCamera::GeneralCamera() {}

CameraPtr GeneralCamera::CreateCamera(Camera::ModelType modelType,
                                      const std::string &CameraName,
                                      const cv::Size imageSize) const {
  switch (modelType) {
  case Camera::FISHEYE: {
    FisheyeCameraPtr camera(new FisheyeCamera);
    FisheyeCamera::Parameters params = camera->getParameters();
    params.CameraName() = CameraName;
    params.imageWidth() = imageSize.width;
    params.imageHeight() = imageSize.height;
    camera->setParameters(params);
    return camera;
  }
  case Camera::PINHOLE: {
    PinholeCameraPtr camera(new PinholeCamera);
    PinholeCamera::Parameters params = camera->getParameters();
    params.CameraName() = CameraName;
    params.imageWidth() = imageSize.width;
    params.imageHeight() = imageSize.height;
    camera->setParameters(params);
    return camera;
  }
  default: {
  }
  }
}