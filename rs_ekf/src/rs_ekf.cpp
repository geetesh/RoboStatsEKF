#include "../include/ekf.h"

EKF::EKF()
{
    ros::NodeHandle nh("~");
    nh.param("odom_frame_id_",odom_frame_id_,std::string("/world"));
    nh.param("imu_frame_id_",imu_frame_id_,std::string("/imu"));
    nh.param("stereo_frame_id_",stereo_frame_id_,std::string("/camera"));
    nh.param("imu_topic",imu_topic,std::string("/imu_topic"));
    nh.param("vo_topic",vo_topic,std::string("/vo/odometry"));
    nh.param("publish_tf", publish_tf ,true);
    nh.param("publish_pose", publish_pose ,true);
    nh.param("accel_cov_times", accel_cov_times, 1000.0);
    nh.param("gryo_cov_times", gryo_cov_times ,1.0);
//    nh.param("vo_correction_dof", vo_correction_dof, 9);
    nh.param("use_vo", use_vo ,true);
    nh.param("calculate_gravity", calculate_gravity, true);

    odom_pub_= local_nh.advertise<nav_msgs::Odometry>("epson_odometry", 10);
    pose_pub_= local_nh.advertise<geometry_msgs::PoseStamped>("epson_pose", 10);

    ros::TransportHints ros_transport_hints = ros::TransportHints().udp().tcpNoDelay();
    imu_sub_ = local_nh.subscribe<sensor_msgs::Imu>(imu_topic, 100, &EKF::imucallback, this, ros_transport_hints);

    if (use_vo){
       vo_sub_ = local_nh.subscribe<nav_msgs::Odometry>(vo_topic, 100, &EKF::vocallback, this, ros_transport_hints);
    }

    if(calculate_gravity){
  gravity.setZero();
    }
    else{
      gravity<< 0, 0, 9.8;
    }
}

void EKF::imucallback(const sensor_msgs::Imu::ConstPtr &imu_msg)
{
  reset_state(cur_state);
  cur_state.msgtime = imu_msg->header.stamp.toSec();
  cur_state.sensor_type = IMU;
  accelerationFromImuMsg(imu_msg, cur_state.accel);
  angularRateFromImuMsg(imu_msg, cur_state.gyro);
  quaternionFromImuMsg(imu_msg, cur_state.ahrs);
}

void EKF::vocallback(const nav_msgs::Odometry::ConstPtr& vo_msg)
{
  reset_state(cur_state);
  cur_state.msgtime = vo_msg->header.stamp.toSec();
    cur_state.sensor_type = VO;
    positionFromOdomMsg(vo_msg, cur_state.vo_pos);
    quaternionFromOdomMsg(vo_msg, cur_state.vo_ahrs);
    posecovFromOdomMsg(vo_msg, cur_state.vo_cov);
    velocityFromOdomMsg(vo_msg, cur_state.vo_vel);
    vo_CtoI(cur_state);
}

void EKF::Compute_Jacobians(const double &dt)
{
   ekf_A.setZero(); //reset Jacobian A
   Matrix_21X21 J;
   J.setZero();
   J.block<3,3>(0,3)  = -last_state.pose.linear() * Hat(last_state.v_b);
   J.block<3,3>(0,6)  =  last_state.pose.linear();
   J.block<3,3>(3,3)  = -Hat(last_state.gyro - last_state.bg);
   J.block<3,3>(3,9)  = -I_3;
   J.block<3,3>(6,3)  =  Hat(last_state.pose.linear().transpose() * gravity);
   J.block<3,3>(6,6)  = -Hat(last_state.gyro - last_state.bg);//
   J.block<3,3>(6,9)  = -Hat(last_state.v_b);//
   J.block<3,3>(6,12) = -I_3;
   ekf_A = I_21 + J * dt;
   ekf_B.setZero(); //reset Jacobian B
   ekf_B.block<3,3>(3,0)  = -I_3;
   ekf_B.block<3,3>(6,0)  = -Hat(last_state.v_b);
   ekf_B.block<3,3>(6,3)  = -I_3;
   ekf_B.block<3,3>(9,6)  =  I_3;
   ekf_B.block<3,3>(12,9) =  I_3;
   ekf_B *= dt;
}

void EKF::reset_state(state_info &state)
{
    state.msgtime = 0.f;
    state.sensor_type = UNKNOWN;
    state.p_w_i.setZero();
    state.q_w_i.setIdentity();
    state.pose.setIdentity();
    state.v_b.setZero();
    state.bg.setZero();
    state.ba.setZero();
    state.state_cov.setZero();
    state.accel.setZero();
    state.gyro.setZero();
    state.vo_pos.setZero();
    state.vo_ahrs.setIdentity();
    state.vo_cov.setZero();
    state.delayed_state.setIdentity();
    state.baro_meas = 0.0;
}

void EKF::vo_CtoI(state_info &state)
{
    Matrix_3X3 Rc, Ri, Ric;
    Eigen::Vector3d tc, ti, tic;
    Rc  = state.vo_ahrs.toRotationMatrix();
    Ric = Tic.linear();
    tc  = state.vo_pos;
    tic = Tic.translation();
    Ri = Ric * Rc * Ric.transpose();
    ti = (I_3 - Ri) * tic + Ric * tc;
    Eigen::Quaterniond temp(Ri);
    temp.normalize();
    state.vo_pos = ti;
    state.vo_ahrs = temp;
}

void EKF::imu_ekf_proc(const double &dt)
{
    Eigen::Vector3d real_w = last_state.gyro - last_state.bg;
    Eigen::Vector4d dq_dt = 0.5 * Omega(real_w) * last_state.q_w_i.coeffs();
    Eigen::Quaterniond qwi(last_state.q_w_i.coeffs() + dt * dq_dt);
    qwi.normalize();
    cur_state.q_w_i = qwi;
    if(cur_state.sensor_type != IMU){
       cur_state.gyro  = last_state.gyro;
       cur_state.accel = last_state.accel;
    }
    Eigen::Vector3d real_accel = last_state.accel - last_state.ba + last_state.q_w_i.inverse() * gravity;
    cur_state.v_b = last_state.v_b + real_accel * dt - Hat(real_w) * last_state.v_b * dt;
    cur_state.p_w_i = last_state.p_w_i + last_state.q_w_i * last_state.v_b * dt;
    cur_state.ba = last_state.ba;
    cur_state.bg = last_state.bg;
    cur_state.pose.linear() = cur_state.q_w_i.toRotationMatrix();
    cur_state.pose.translation() = cur_state.p_w_i;
    cur_state.delayed_state = last_state.delayed_state;
    Compute_Jacobians(dt);
    ekf_P = last_state.state_cov;
    Matrix_21X21 P_ = ekf_P;
    ekf_P = ekf_A * P_ * ekf_A.transpose() + ekf_B * ekf_Q * ekf_B.transpose();
    if(( ekf_P.diagonal().array() <= 0).any() ){
        ROS_ERROR_STREAM("[Prediction] : ERROR!!! The state covariance P is negative definition now !!!" );
    return;
    }
    cur_state.state_cov = ekf_P;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RS_EKF");
       ros::start();
       EKF rs_ekf;
       ros::spin();
       ros::shutdown();
    return 0;
}
