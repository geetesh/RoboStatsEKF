#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <tf/LinearMath/Transform.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

typedef Eigen::Matrix<double,  6,  1> Vector_6X1d;  //measurement
typedef Eigen::Matrix<double, 21,  1> Vector_21X1d; //new ekf state X

typedef Eigen::Matrix<double, 12, 12> Matirx_12X12; //imu noise cov
typedef Eigen::Matrix<double,  3,  3> Matrix_3X3;
typedef Eigen::Matrix<double,  6,  6> Matrix_6X6;   // measurement cov
typedef Eigen::Matrix<double, 21, 21> Matrix_21X21; //state covariance and jacobian A
typedef Eigen::Matrix<double, 21, 12> Matrix_21x12; //noise Jacobian B
typedef Eigen::Matrix<double,  3, 21> Matrix_3x21;  //measurement matrix H
typedef Eigen::Matrix<double,  6, 21> Matrix_6x21;  //measurement matrix H

enum Sensor{UNKNOWN = 0, IMU = 1, BARO = 2, VO = 3 };

typedef struct{
   double msgtime;
   Sensor sensor_type;

   //state information
   Eigen::Vector3d p_w_i;
   Eigen::Quaterniond q_w_i;
   Eigen::Isometry3d pose;
   Eigen::Vector3d v_b;
   Eigen::Vector3d ba;
   Eigen::Vector3d bg;
   //state covariance
   Matrix_21X21 state_cov;
   // delayed state
   Eigen::Isometry3d delayed_state;

   // IMU measuement
   Eigen::Vector3d accel;
   Eigen::Vector3d gyro;
   Eigen::Quaterniond ahrs; //arhs orientation in world frame

   // vo measurement
   Eigen::Vector3d vo_pos;     //delta position w.r.t last ref-frame
   Eigen::Quaterniond vo_ahrs; //delta orientation w.r.t last ref-frame
   Matrix_6X6 vo_cov;
   Eigen::Vector3d vo_vel;     //delta position w.r.t last ref-frame

   // barometer measurement
   double baro_meas;
} state_info;

// define some identity matrixs
const Matrix_21X21 I_21 = Eigen::MatrixXd::Identity(21,21);
const Matrix_3X3 I_3 = Eigen::MatrixXd::Identity(3,3);

class EKF
{
public:
    EKF();
    void predict();
    void update();

private:
    std::string imu_topic, vo_topic;
    ros::NodeHandle local_nh;
    ros::Publisher odom_pub_, pose_pub_;
    ros::Subscriber imu_sub_, vo_sub_;

    //tf-related
    tf::TransformBroadcaster tf_broadcaster_;
    tf::TransformListener tf_listener_;
    Eigen::Isometry3d Tic; //the calibration from stereo to imu
    std::string imu_frame_id_, odom_frame_id_, stereo_frame_id_;

    //callback
    void imucallback(const sensor_msgs::Imu::ConstPtr & imu_msg);
    void vocallback(const nav_msgs::Odometry::ConstPtr& vo_msg);

    // imu-ekf related
    double last_imu_time;
    state_info last_state,cur_state;
    Vector_21X1d ekf_X; // ekf state vector [position , delta_angle, velocity, bias_gryo, bias_accl];
    Matrix_21X21 ekf_P; //ekf_covariance P
    Matirx_12X12 ekf_Q; //noise_covariance Q
    Matrix_21X21 ekf_A; //Jacobian for IMU system dynamic
    Matrix_21x12 ekf_B; //Jacobian B for noise sysem dynamic

    void Compute_Jacobians(const double &dt);//calcalate Jacobians
    void imu_ekf_proc(const double &dt);//state propagation process
    void set_ekf_from_state(state_info &state);
    void update_state_by_ekf(state_info &state);
    void reset_state(state_info &state);
    void Initialize_ekf_covaiance();
    void vo_CtoI(state_info &state);

    //other
    bool publish_tf, publish_pose, use_vo,calculate_gravity;
    Eigen::Vector3d gravity;
    double accel_cov_times, gryo_cov_times;

    // get information from IMU and DJI messages
    static inline void angularRateFromImuMsg( const sensor_msgs::Imu::ConstPtr & msg, Eigen::Vector3d & v ){
       v.x() = msg->angular_velocity.x;
       v.y() = -msg->angular_velocity.y;
       v.z() = -msg->angular_velocity.z;
    }
    static inline void accelerationFromImuMsg( const sensor_msgs::Imu::ConstPtr & msg, Eigen::Vector3d & v ){
       v.x() = msg->linear_acceleration.x;
       v.y() = -msg->linear_acceleration.y;
       v.z() = -msg->linear_acceleration.z;
    }
    static inline void quaternionFromImuMsg( const sensor_msgs::Imu::ConstPtr & msg, Eigen::Quaterniond & q ){
       q.x() = msg->orientation.x;
       q.y() = msg->orientation.y;
       q.z() = msg->orientation.z;
       q.w() = msg->orientation.w;
    }
    static inline void positionFromOdomMsg( const nav_msgs::Odometry::ConstPtr & msg, Eigen::Vector3d & v ){
       v.x() = msg->pose.pose.position.x;
       v.y() = msg->pose.pose.position.y;
       v.z() = msg->pose.pose.position.z;
    }

    static inline void velocityFromOdomMsg( const nav_msgs::Odometry::ConstPtr & msg, Eigen::Vector3d & v ){
       v.x() = msg->twist.twist.linear.x;
       v.y() = msg->twist.twist.linear.y;
       v.z() = msg->twist.twist.linear.z;
    }
    static inline void quaternionFromOdomMsg( const nav_msgs::Odometry::ConstPtr & msg, Eigen::Quaterniond & q ){
       q.x() = msg->pose.pose.orientation.x;
       q.y() = msg->pose.pose.orientation.y;
       q.z() = msg->pose.pose.orientation.z;
       q.w() = msg->pose.pose.orientation.w;
       q.normalize();
    }
    static inline void posecovFromOdomMsg( const nav_msgs::Odometry::ConstPtr & msg, Matrix_6X6 & m ){
       for( int row = 0; row < 6; row ++ ){
          for( int col = 0; col < 6; col ++ ){
        m( row, col ) =  msg->pose.covariance[ row * 6 + col ];
          }
       }
    }
    static inline void velcovFromOdomMsg( const nav_msgs::Odometry::ConstPtr & msg, Matrix_6X6 & m ){
       for( int row = 0; row < 6; row ++ ){
          for( int col = 0; col < 6; col ++ ){
        m( row, col ) =  msg->twist.covariance[ row * 6 + col ];}
       }
    }
    static inline void copyCov(int src, int dst, Matrix_21X21 & P){
           P.row(dst) = P.row(src);
           P.col(dst) = P.col(src);
    }
    static inline void RPbyGravity(const Eigen::Vector3d &g, double& pitch, double& roll){
           roll = atan2(-g.y(),-g.z()); //(-pi,pi)
           pitch = -asin(-g.x()/g.norm());//(-pi/2,pi/2)
    }

    Eigen::Matrix3d Hat(const Eigen::Vector3d w)
    {
        Eigen::Matrix3d W;
        W <<   0.0, -w(2),  w(1),
              w(2),   0.0, -w(0),
             -w(1),  w(0),   0.0;
        return W;
    }

    Eigen::Matrix4d Omega(const Eigen::Vector3d w)
    {
       Eigen::Matrix3d skew_w = Hat(w);
       Eigen::Matrix4d omega_w;
       omega_w << -skew_w, 		w,
              -w.transpose(), 	0;
       return omega_w;
    }


};
