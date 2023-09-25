#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <ceres/ceres.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <vector>

class ExtrinsicOnlyFactor : public ceres::SizedCostFunction<3, 6> {
public:
  ExtrinsicOnlyFactor(const Eigen::Vector3d &_pts_i,
                      const Eigen::Vector3d &_pts_j);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d pts_i, pts_j;
};

// class QuaternionFactor : public ceres::SizedCostFunction<4, 4> {
// public:
//   QuaternionFactor(const Eigen::Quaterniond &Q_cam_,
//                    const Eigen::Quaterniond &Q_imu_);
//   virtual bool Evaluate(double const *const *parameters, double *residuals,
//                         double **jacobians) const;

//   Eigen::Vector3d Q_cam, Q_imu;
// };

class EucmFactor : public ceres::SizedCostFunction<2, 6, 2, 4> {
public:
  EucmFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d pts_i, pts_j;
};

class Camera;

typedef boost::shared_ptr<Camera> CameraPtr;
typedef boost::shared_ptr<const Camera> CameraConstPtr;

class Camera {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum ModelType { FISHEYE, MEI, PINHOLE, SCARAMUZZA };

  class Parameters {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Parameters(ModelType modelType);

    Parameters(ModelType modelType, const std::string &CameraName, int w,
               int h);

    ModelType &modelType(void);
    std::string &CameraName(void);
    int &imageWidth(void);
    int &imageHeight(void);

    ModelType modelType(void) const;
    const std::string &CameraName(void) const;
    int imageWidth(void) const;
    int imageHeight(void) const;

    int nIntrinsics(void) const;

    virtual bool readFromYamlFile(const std::string &filename) = 0;
    virtual void writeToYamlFile(const std::string &filename) const = 0;

  protected:
    ModelType m_modelType;
    int m_nIntrinsics;
    std::string mCameraName;
    int m_imageWidth;
    int m_imageHeight;
  };

  virtual ModelType modelType(void) const = 0;
  virtual const std::string &CameraName(void) const = 0;
  virtual int imageWidth(void) const = 0;
  virtual int imageHeight(void) const = 0;

  virtual cv::Mat &mask(void);
  virtual const cv::Mat &mask(void) const;

  virtual void estimateIntrinsics(
      const cv::Size &boardSize,
      const std::vector<std::vector<cv::Point3f>> &objectPoints,
      const std::vector<std::vector<cv::Point2f>> &imagePoints) = 0;
  virtual void estimateExtrinsics(const std::vector<cv::Point3f> &objectPoints,
                                  const std::vector<cv::Point2f> &imagePoints,
                                  cv::Mat &rvec, cv::Mat &tvec) const;

  // Lift points from the image plane to the sphere
  virtual void liftSphere(const Eigen::Vector2d &p,
                          Eigen::Vector3d &P) const = 0;
  //%output P

  // Lift points from the image plane to the projective space
  virtual void liftProjective(const Eigen::Vector2d &p,
                              Eigen::Vector3d &P) const = 0;
  //%output P

  // Projects 3D points to the image plane (Pi function)
  virtual void spaceToPlane(const Eigen::Vector3d &P,
                            Eigen::Vector2d &p) const = 0;
  //%output p

  // Projects 3D points to the image plane (Pi function)
  // and calculates jacobian
  // virtual void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
  //                          Eigen::Matrix<double,2,3>& J) const = 0;
  //%output p
  //%output J

  virtual void undistToPlane(const Eigen::Vector2d &p_u,
                             Eigen::Vector2d &p) const = 0;
  //%output p

  // virtual void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale
  // = 1.0) const = 0;
  virtual cv::Mat
  initUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2, float fx = -1.0f,
                          float fy = -1.0f, cv::Size imageSize = cv::Size(0, 0),
                          float cx = -1.0f, float cy = -1.0f,
                          cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const = 0;

  virtual int parameterCount(void) const = 0;

  virtual void readParameters(const std::vector<double> &parameters) = 0;
  virtual void writeParameters(std::vector<double> &parameters) const = 0;

  virtual void writeParametersToYamlFile(const std::string &filename) const = 0;

  virtual std::string parametersToString(void) const = 0;

  /**
   * \brief Calculates the reprojection distance between points
   *
   * \param P1 first 3D point coordinates
   * \param P2 second 3D point coordinates
   * \return euclidean distance in the plane
   */
  double ReprojectionError(const Eigen::Vector3d &P1,
                           const Eigen::Vector3d &P2) const;

  double
  ReprojectionError(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                    const std::vector<std::vector<cv::Point2f>> &imagePoints,
                    const std::vector<cv::Mat> &rvecs,
                    const std::vector<cv::Mat> &tvecs,
                    cv::OutputArray perViewErrors = cv::noArray()) const;

  double ReprojectionError(const Eigen::Vector3d &P,
                           const Eigen::Quaterniond &camera_q,
                           const Eigen::Vector3d &camera_t,
                           const Eigen::Vector2d &observed_p) const;

  void projectPoints(const std::vector<cv::Point3f> &objectPoints,
                     const cv::Mat &rvec, const cv::Mat &tvec,
                     std::vector<cv::Point2f> &imagePoints) const;

protected:
  cv::Mat m_mask;
};
class PinholeCamera : public Camera {
public:
  class Parameters : public Camera::Parameters {
  public:
    Parameters();
    Parameters(const std::string &CameraName, int w, int h, double k1,
               double k2, double p1, double p2, double fx, double fy, double cx,
               double cy);

    double &k1(void);
    double &k2(void);
    double &p1(void);
    double &p2(void);
    double &fx(void);
    double &fy(void);
    double &cx(void);
    double &cy(void);

    double xi(void) const;
    double k1(void) const;
    double k2(void) const;
    double p1(void) const;
    double p2(void) const;
    double fx(void) const;
    double fy(void) const;
    double cx(void) const;
    double cy(void) const;

    bool readFromYamlFile(const std::string &filename);
    void writeToYamlFile(const std::string &filename) const;

    Parameters &operator=(const Parameters &other);
    friend std::ostream &operator<<(std::ostream &out,
                                    const Parameters &params);

  private:
    double m_k1;
    double m_k2;
    double m_p1;
    double m_p2;
    double m_fx;
    double m_fy;
    double m_cx;
    double m_cy;
  };

  PinholeCamera();

  /**
   * \brief Constructor from the projection model parameters
   */
  PinholeCamera(const std::string &CameraName, int imageWidth, int imageHeight,
                double k1, double k2, double p1, double p2, double fx,
                double fy, double cx, double cy);
  /**
   * \brief Constructor from the projection model parameters
   */
  PinholeCamera(const Parameters &params);

  Camera::ModelType modelType(void) const;
  const std::string &CameraName(void) const;
  int imageWidth(void) const;
  int imageHeight(void) const;

  void
  estimateIntrinsics(const cv::Size &boardSize,
                     const std::vector<std::vector<cv::Point3f>> &objectPoints,
                     const std::vector<std::vector<cv::Point2f>> &imagePoints);

  // Lift points from the image plane to the sphere
  virtual void liftSphere(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;
  //%output P

  // Lift points from the image plane to the projective space
  void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;
  //%output P

  // Projects 3D points to the image plane (Pi function)
  void spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const;
  //%output p

  // Projects 3D points to the image plane (Pi function)
  // and calculates jacobian
  void spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p,
                    Eigen::Matrix<double, 2, 3> &J) const;
  //%output p
  //%output J

  void undistToPlane(const Eigen::Vector2d &p_u, Eigen::Vector2d &p) const;
  //%output p

  template <typename T>
  static void spaceToPlane(const T *const params, const T *const q,
                           const T *const t, const Eigen::Matrix<T, 3, 1> &P,
                           Eigen::Matrix<T, 2, 1> &p);

  void distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u) const;
  void distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u,
                  Eigen::Matrix2d &J) const;

  void initUndistortMap(cv::Mat &map1, cv::Mat &map2,
                        double fScale = 1.0) const;
  cv::Mat
  initUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2, float fx = -1.0f,
                          float fy = -1.0f, cv::Size imageSize = cv::Size(0, 0),
                          float cx = -1.0f, float cy = -1.0f,
                          cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

  int parameterCount(void) const;

  const Parameters &getParameters(void) const;
  void setParameters(const Parameters &parameters);

  void readParameters(const std::vector<double> &parameterVec);
  void writeParameters(std::vector<double> &parameterVec) const;

  void writeParametersToYamlFile(const std::string &filename) const;

  std::string parametersToString(void) const;

private:
  Parameters mParameters;

  double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
  bool m_noDistortion;
};

typedef boost::shared_ptr<PinholeCamera> PinholeCameraPtr;
typedef boost::shared_ptr<const PinholeCamera> PinholeCameraConstPtr;

template <typename T>
void PinholeCamera::spaceToPlane(const T *const params, const T *const q,
                                 const T *const t,
                                 const Eigen::Matrix<T, 3, 1> &P,
                                 Eigen::Matrix<T, 2, 1> &p) {
  T P_w[3];
  P_w[0] = T(P(0));
  P_w[1] = T(P(1));
  P_w[2] = T(P(2));

  // Convert quaternion from Eigen convention (x, y, z, w)
  // to Ceres convention (w, x, y, z)
  T q_ceres[4] = {q[3], q[0], q[1], q[2]};

  T P_c[3];
  ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

  P_c[0] += t[0];
  P_c[1] += t[1];
  P_c[2] += t[2];

  // project 3D object point to the image plane
  T k1 = params[0];
  T k2 = params[1];
  T p1 = params[2];
  T p2 = params[3];
  T fx = params[4];
  T fy = params[5];
  T alpha = T(0); // cameraParams.alpha();
  T cx = params[6];
  T cy = params[7];

  // Transform to model plane
  T u = P_c[0] / P_c[2];
  T v = P_c[1] / P_c[2];

  T rho_sqr = u * u + v * v;
  T L = T(1.0) + k1 * rho_sqr + k2 * rho_sqr * rho_sqr;
  T du = T(2.0) * p1 * u * v + p2 * (rho_sqr + T(2.0) * u * u);
  T dv = p1 * (rho_sqr + T(2.0) * v * v) + T(2.0) * p2 * u * v;

  u = L * u + du;
  v = L * v + dv;
  p(0) = fx * (u + alpha * v) + cx;
  p(1) = fy * v + cy;
}

class FisheyeCamera : public Camera {
public:
  class Parameters : public Camera::Parameters {
  public:
    Parameters();
    Parameters(const std::string &CameraName, int w, int h, double k2,
               double k3, double k4, double k5, double mu, double mv, double u0,
               double v0);

    double &k2(void);
    double &k3(void);
    double &k4(void);
    double &k5(void);
    double &mu(void);
    double &mv(void);
    double &u0(void);
    double &v0(void);

    double k2(void) const;
    double k3(void) const;
    double k4(void) const;
    double k5(void) const;
    double mu(void) const;
    double mv(void) const;
    double u0(void) const;
    double v0(void) const;

    bool readFromYamlFile(const std::string &filename);
    void writeToYamlFile(const std::string &filename) const;

    Parameters &operator=(const Parameters &other);
    friend std::ostream &operator<<(std::ostream &out,
                                    const Parameters &params);

  private:
    // projection
    double m_k2;
    double m_k3;
    double m_k4;
    double m_k5;

    double m_mu;
    double m_mv;
    double m_u0;
    double m_v0;
  };

  FisheyeCamera();

  /**
   * \brief Constructor from the projection model parameters
   */
  FisheyeCamera(const std::string &CameraName, int imageWidth, int imageHeight,
                double k2, double k3, double k4, double k5, double mu,
                double mv, double u0, double v0);
  /**
   * \brief Constructor from the projection model parameters
   */
  FisheyeCamera(const Parameters &params);

  Camera::ModelType modelType(void) const;
  const std::string &CameraName(void) const;
  int imageWidth(void) const;
  int imageHeight(void) const;

  void
  estimateIntrinsics(const cv::Size &boardSize,
                     const std::vector<std::vector<cv::Point3f>> &objectPoints,
                     const std::vector<std::vector<cv::Point2f>> &imagePoints);

  // Lift points from the image plane to the sphere
  virtual void liftSphere(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;
  //%output P

  // Lift points from the image plane to the projective space
  void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;
  //%output P

  // Projects 3D points to the image plane (Pi function)
  void spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const;
  //%output p

  // Projects 3D points to the image plane (Pi function)
  // and calculates jacobian
  void spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p,
                    Eigen::Matrix<double, 2, 3> &J) const;
  //%output p
  //%output J

  void undistToPlane(const Eigen::Vector2d &p_u, Eigen::Vector2d &p) const;
  //%output p

  template <typename T>
  static void spaceToPlane(const T *const params, const T *const q,
                           const T *const t, const Eigen::Matrix<T, 3, 1> &P,
                           Eigen::Matrix<T, 2, 1> &p);

  void initUndistortMap(cv::Mat &map1, cv::Mat &map2,
                        double fScale = 1.0) const;
  cv::Mat
  initUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2, float fx = -1.0f,
                          float fy = -1.0f, cv::Size imageSize = cv::Size(0, 0),
                          float cx = -1.0f, float cy = -1.0f,
                          cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

  int parameterCount(void) const;

  const Parameters &getParameters(void) const;
  void setParameters(const Parameters &parameters);

  void readParameters(const std::vector<double> &parameterVec);
  void writeParameters(std::vector<double> &parameterVec) const;

  void writeParametersToYamlFile(const std::string &filename) const;

  std::string parametersToString(void) const;

private:
  template <typename T> static T r(T k2, T k3, T k4, T k5, T theta);

  void fitOddPoly(const std::vector<double> &x, const std::vector<double> &y,
                  int n, std::vector<double> &coeffs) const;

  void backprojectSymmetric(const Eigen::Vector2d &p_u, double &theta,
                            double &phi) const;

  void CalCircle(const std::vector<cv::Point2d> &points, double &centerX,
                 double &centerY, double &radius);
  std::vector<cv::Point2d> CircleIntersection(double x1, double y1, double r1,
                                              double x2, double y2, double r2);

  Parameters mParameters;
  template <class T> const T square(const T &x) { return x * x; }
  double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
};

typedef boost::shared_ptr<FisheyeCamera> FisheyeCameraPtr;
typedef boost::shared_ptr<const FisheyeCamera> FisheyeCameraConstPtr;

template <typename T> T FisheyeCamera::r(T k2, T k3, T k4, T k5, T theta) {
  // k1 = 1
  return theta + k2 * theta * theta * theta +
         k3 * theta * theta * theta * theta * theta +
         k4 * theta * theta * theta * theta * theta * theta * theta +
         k5 * theta * theta * theta * theta * theta * theta * theta * theta *
             theta;
}

template <typename T>
void FisheyeCamera::spaceToPlane(const T *const params, const T *const q,
                                 const T *const t,
                                 const Eigen::Matrix<T, 3, 1> &P,
                                 Eigen::Matrix<T, 2, 1> &p) {
  T P_w[3];
  P_w[0] = T(P(0));
  P_w[1] = T(P(1));
  P_w[2] = T(P(2));

  // Convert quaternion from Eigen convention (x, y, z, w)
  // to Ceres convention (w, x, y, z)
  T q_ceres[4] = {q[3], q[0], q[1], q[2]};

  T P_c[3];
  ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

  P_c[0] += t[0];
  P_c[1] += t[1];
  P_c[2] += t[2];

  // project 3D object point to the image plane;
  T k2 = params[0];
  T k3 = params[1];
  T k4 = params[2];
  T k5 = params[3];
  T mu = params[4];
  T mv = params[5];
  T u0 = params[6];
  T v0 = params[7];

  T len = sqrt(P_c[0] * P_c[0] + P_c[1] * P_c[1] + P_c[2] * P_c[2]);
  T theta = acos(P_c[2] / len);
  T phi = atan2(P_c[1], P_c[0]);

  Eigen::Matrix<T, 2, 1> p_u =
      r(k2, k3, k4, k5, theta) * Eigen::Matrix<T, 2, 1>(cos(phi), sin(phi));

  p(0) = mu * p_u(0) + u0;
  p(1) = mv * p_u(1) + v0;
}

class CostFunction {
public:
  CostFunction() {}
  static boost::shared_ptr<CostFunction> instance(void);
  ceres::CostFunction *
  CreateCostFunction(const CameraConstPtr &camera,
                     const Eigen::Vector3d &observed_P,
                     const Eigen::Vector2d &observed_p) const;

  ceres::CostFunction *CreateCostFunction(
      const CameraConstPtr &cameraL, const CameraConstPtr &cameraR,
      const Eigen::Vector3d &observed_P, const Eigen::Vector2d &observed_p_l,
      const Eigen::Vector2d &observed_p_r) const;

private:
  static boost::shared_ptr<CostFunction> CostFunctionInstance;
};

class CameraCalibration {
public:
  CameraCalibration();

  CameraCalibration(Camera::ModelType modelType, const std::string &CameraName,
                    const cv::Size &imageSize, const cv::Size &boardSize,
                    float squareSize);

  CameraCalibration(Camera::ModelType modelTypeLeft,
                    Camera::ModelType modelTypeRight,
                    const std::string &CameraNameLeft,
                    const std::string &CameraNameRight,
                    const cv::Size &imageSizeLeft,
                    const cv::Size &imageSizeRight, const cv::Size &boardSize,
                    float squareSize);

  void clear(void);

  bool findChessboardCorners(const cv::Mat &image, const cv::Size &patternSize,
                             std::vector<cv::Point2f> &corners);

  void addChessboardData(std::vector<std::vector<cv::Point3f>> &world_pts,
                         std::vector<std::vector<cv::Point2f>> &img_pts);

  void
  addStereoChessboardData(std::vector<std::vector<cv::Point3f>> &world_pts,
                          std::vector<std::vector<cv::Point2f>> &img_pts_left,
                          std::vector<std::vector<cv::Point2f>> &img_pts_right);

  bool calibrate(void);
  bool calibrateStereo(void);

  int sampleCount(void) const;
  std::vector<std::vector<cv::Point2f>> &imagePoints(void);
  const std::vector<std::vector<cv::Point2f>> &imagePoints(void) const;
  std::vector<std::vector<cv::Point3f>> &scenePoints(void);
  const std::vector<std::vector<cv::Point3f>> &scenePoints(void) const;
  CameraPtr &camera(void);
  const CameraConstPtr camera(void) const;

  Eigen::Matrix2d &measurementCovariance(void);
  const Eigen::Matrix2d &measurementCovariance(void) const;

  cv::Mat &cameraPoses(void);
  const cv::Mat &cameraPoses(void) const;

  void drawResults(std::vector<cv::Mat> &images) const;

  void writeParams(const std::string &filename) const;

  bool writeChessboardData(const std::string &filename) const;
  bool readChessboardData(const std::string &filename);

  void setVerbose(bool verbose);

  bool CalibrationExecute(CameraPtr &camera,
                          std::vector<std::vector<cv::Point2f>> &image_points,
                          std::vector<cv::Mat> &rvecs,
                          std::vector<cv::Mat> &tvecs) const;

  bool StereoCalibrationExecute(
      CameraPtr &camera_left, CameraPtr &camera_right,
      std::vector<std::vector<cv::Point2f>> &image_points_left,
      std::vector<std::vector<cv::Point2f>> &image_points_right,
      std::vector<cv::Mat> &rvecs_left, std::vector<cv::Mat> &tvecs_left,
      std::vector<cv::Mat> &rvecs_right,
      std::vector<cv::Mat> &tvecs_right) const;

  void MonoOptimizer(CameraPtr &camera,
                     std::vector<std::vector<cv::Point2f>> &image_points,
                     std::vector<cv::Mat> &rvecs,
                     std::vector<cv::Mat> &tvecs) const;

  void
  StereoOptimizer(CameraPtr &camera_left, CameraPtr &camera_right,
                  std::vector<std::vector<cv::Point2f>> &image_points_left,
                  std::vector<std::vector<cv::Point2f>> &image_points_right,
                  std::vector<cv::Mat> &rvecs_left,
                  std::vector<cv::Mat> &tvecs_left,
                  std::vector<cv::Mat> &rvecs_right,
                  std::vector<cv::Mat> &tvecs_right) const;

  template <typename T> void ReadData(std::ifstream &ifs, T &data) const;

  template <typename T> void WriteData(std::ofstream &ofs, T data) const;

  cv::Size mBoardSize;
  float mSquareSize;

  // MONO Calibration
  CameraPtr mCamera;
  cv::Mat mCameraPoses;
  std::vector<std::vector<cv::Point2f>> mImagePoints;
  std::vector<std::vector<cv::Point3f>> mScenePoints;

  // STEREO Calibration
  CameraPtr mCamera_left;
  CameraPtr mCamera_right;
  cv::Mat mCameraPoses_left;
  cv::Mat mCameraPoses_right;
  std::vector<std::vector<cv::Point2f>> mImagePoints_left;
  std::vector<std::vector<cv::Point2f>> mImagePoints_right;
  cv::Mat ExtrinsicR;
  cv::Mat ExtrinsicT;
  Eigen::Matrix2d mMeasurementCovariance;

  bool mVerBose;
};

class Transform {
public:
  Transform();
  Transform(const Eigen::Matrix4d &H);

  Eigen::Quaterniond &rotation(void);
  const Eigen::Quaterniond &rotation(void) const;
  double *rotationData(void);
  const double *const rotationData(void) const;

  Eigen::Vector3d &translation(void);
  const Eigen::Vector3d &translation(void) const;
  double *translationData(void);
  const double *const translationData(void) const;

  Eigen::Matrix4d toMatrix(void) const;

  template <typename T>
  Eigen::Quaternion<T>
  AngleAxisToQuaternion(const Eigen::Matrix<T, 3, 1> &rvec);

  template <typename T>
  Eigen::Matrix<T, 3, 3>
  AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1> &rvec);

private:
  Eigen::Quaterniond m_q;
  Eigen::Vector3d m_t;
};

class EigenQuaternionParameterization : public ceres::LocalParameterization {
public:
  virtual ~EigenQuaternionParameterization() {}
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 4; }
  virtual int LocalSize() const { return 3; }

private:
  template <typename T>
  void EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const;
};

template <typename T>
void EigenQuaternionParameterization::EigenQuaternionProduct(const T z[4],
                                                             const T w[4],
                                                             T zw[4]) const {
  zw[0] = z[3] * w[0] + z[0] * w[3] + z[1] * w[2] - z[2] * w[1];
  zw[1] = z[3] * w[1] - z[0] * w[2] + z[1] * w[3] + z[2] * w[0];
  zw[2] = z[3] * w[2] + z[0] * w[1] - z[1] * w[0] + z[2] * w[3];
  zw[3] = z[3] * w[3] - z[0] * w[0] - z[1] * w[1] - z[2] * w[2];
}

class GeneralCamera {
public:
  GeneralCamera();
  static boost::shared_ptr<GeneralCamera> instance(void);

  CameraPtr CreateCamera(Camera::ModelType modelType,
                         const std::string &CameraName,
                         const cv::Size imageSize) const;

private:
  static boost::shared_ptr<GeneralCamera> GeneralCameraInstanceinstance;
};
