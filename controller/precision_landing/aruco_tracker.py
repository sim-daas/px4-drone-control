#!/usr/bin/env python3
# pyrefly: ignore [missing-import]
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String, Bool
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
import time

class PerceptionTracker(Node):
    def __init__(self):
        super().__init__('perception_tracker')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.is_processing = False
        
        # Initialize YOLOv8m
        self.get_logger().info("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8m.pt', )
        self.get_logger().info("YOLO loaded.")
        
        # Initialize ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.6  # metres (matches SDF)
        self.enable_aruco = False

        # Camera parameters (from SDF if camera_info is not available)
        self.hfov = 2.5 # radians
        self.img_width = 1280.0
        self.img_height = 960.0
        
        # Fallback Camera Matrix based on FOV (in case CameraInfo is missing)
        # fx = W / (2 * tan(HFOV/2))
        f_fallback = self.img_width / (2.0 * np.tan(self.hfov / 2.0))
        self.camera_matrix = np.array([[f_fallback, 0, self.img_width/2.0],
                                     [0, f_fallback, self.img_height/2.0],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        
        self.got_camera_info = False
        
        # Drone state
        self.drone_pos = np.zeros(3)
        self.drone_quat = np.array([0.0, 0.0, 0.0, 1.0]) # [x, y, z, w]

        # Platform state estimation (YOLO)
        self.last_yolo_pos_world = None
        self.last_yolo_time = None
        self.yolo_vel_world = np.zeros(3)

        # Platform state estimation (ArUco)
        self.last_aruco_pos_world = None
        self.last_aruco_time = None
        self.aruco_vel_world = np.zeros(3)

        # Subscribers
        self.image_sub = self.create_subscription(CompressedImage, '/camera/compressed', self.image_callback, 1)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        self.info_sub_alt = self.create_subscription(CameraInfo, '/camera/info', self.info_callback, 10) # Alternative topic
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.enable_aruco_sub = self.create_subscription(Bool, '/precision_landing/enable_aruco', self.enable_aruco_callback, 10)

        # Publishers
        self.det_pub = self.create_publisher(String, '/det', 10)
        self.platform_pose_pub = self.create_publisher(PoseStamped, '/precision_landing/platform_pose', 10)
        self.platform_vel_pub = self.create_publisher(TwistStamped, '/precision_landing/platform_velocity', 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, '/precision_landing/debug_image/compressed', 1)

        self.get_logger().info("Perception Tracker initialized")

    def info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.img_width = float(msg.width)
            self.img_height = float(msg.height)
            self.got_camera_info = True
            self.get_logger().info(f"Camera parameters received: {self.camera_matrix.flatten()}")

    def odom_callback(self, msg):
        # PX4 [w, x, y, z] -> SciPy [x, y, z, w]
        self.drone_quat = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]])
        self.drone_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])

    def enable_aruco_callback(self, msg):
        if msg.data and not self.enable_aruco:
            self.get_logger().info("ArUco detection ENABLED.")
            self.enable_aruco = True
        elif not msg.data and self.enable_aruco:
            self.get_logger().info("ArUco detection DISABLED.")
            self.enable_aruco = False

    def project_to_ground(self, nx, ny, drone_pos, drone_quat):
        # nx, ny are normalized coordinates from -1 to 1
        tan_half_hfov = np.tan(self.hfov / 2.0)
        aspect_ratio = self.img_height / self.img_width
        
        # Ray in camera frame
        ray_c = np.array([
            nx * tan_half_hfov,
            ny * aspect_ratio * tan_half_hfov,
            1.0
        ])
        
        # Camera to Body FRD (Assuming X_cam=Y_body, Y_cam=-X_body, Z_cam=Z_body)
        ray_b = np.array([-ray_c[1], ray_c[0], ray_c[2]])
        
        # Body to World NED
        rot_body_world = R.from_quat(drone_quat).as_matrix()
        ray_w = rot_body_world @ ray_b
        
        # Intersect with ground plane Z=0
        if ray_w[2] <= 0:
            # Ray points parallel to or away from ground (pointing up in NED)
            return None
            
        t = -drone_pos[2] / ray_w[2]
        intersection = drone_pos + t * ray_w
        return intersection

    def image_callback(self, msg):
        if self.is_processing:
            return
        self.is_processing = True

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
                
            height, width, _ = cv_image.shape
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # Cache the IMU state exactly when the image arrives (before inference lag!)
            img_drone_pos = self.drone_pos.copy()
            img_drone_quat = self.drone_quat.copy()
            
            # --- 1. YOLO Car Detection ---
            results = self.yolo_model.predict(cv_image, classes=[2, 7], verbose=False, device=0)
            
            det_data = {"detected": False, "boxes": []}
            best_car_pos = None
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    nx = (cx - width / 2.0) / (width / 2.0)
                    ny = (cy - height / 2.0) / (height / 2.0)
                    
                    # Project to ground using the synchronized IMU state
                    world_pos = self.project_to_ground(nx, ny, img_drone_pos, img_drone_quat)
                    
                    if world_pos is not None:
                        # Estimate velocity
                        if self.last_yolo_pos_world is not None and self.last_yolo_time is not None:
                            dt = current_time - self.last_yolo_time
                            if dt > 0:
                                vel = (world_pos - self.last_yolo_pos_world) / dt
                                self.yolo_vel_world = 0.5 * self.yolo_vel_world + 0.5 * vel # Low pass
                        
                        self.last_yolo_pos_world = world_pos
                        self.last_yolo_time = current_time
                        best_car_pos = world_pos
                        
                        det_data["detected"] = True
                        det_data["boxes"].append({
                            "class": cls,
                            "confidence": conf,
                            "nx": nx,
                            "ny": ny,
                            "world_x": world_pos[0],
                            "world_y": world_pos[1],
                            "world_z": world_pos[2],
                            "vel_x": self.yolo_vel_world[0],
                            "vel_y": self.yolo_vel_world[1],
                            "vel_z": self.yolo_vel_world[2]
                        })
                    
                    # Draw YOLO box
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"Car {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            det_msg = String()
            det_msg.data = json.dumps(det_data)
            self.det_pub.publish(det_msg)

            # --- 2. ArUco Marker Detection ---
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

            if ids is not None:
                if self.camera_matrix is not None:
                    marker_points = np.array([[-self.marker_size / 2,  self.marker_size / 2, 0],
                                              [ self.marker_size / 2,  self.marker_size / 2, 0],
                                              [ self.marker_size / 2, -self.marker_size / 2, 0],
                                              [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
                    
                    _, rvec, tvec = cv2.solvePnP(marker_points, corners[0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    rvec = rvec.reshape(3)
                    tvec = tvec.reshape(3)

                    R_cam_marker = R.from_rotvec(rvec).as_matrix()
                    pos_cam_marker = tvec

                    pos_body_marker = np.array([-pos_cam_marker[1], pos_cam_marker[0], pos_cam_marker[2]])
                    rot_body_world = R.from_quat(img_drone_quat).as_matrix()
                    pos_world_marker = rot_body_world @ pos_body_marker + img_drone_pos

                    # Publish Pose
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header
                    pose_msg.header.frame_id = "world"
                    pose_msg.pose.position.x = pos_world_marker[0]
                    pose_msg.pose.position.y = pos_world_marker[1]
                    pose_msg.pose.position.z = pos_world_marker[2]
                    self.platform_pose_pub.publish(pose_msg)

                    # Velocity Estimation
                    if self.last_aruco_pos_world is not None:
                        dt = current_time - self.last_aruco_time
                        if dt > 0:
                            vel = (pos_world_marker - self.last_aruco_pos_world) / dt
                            self.aruco_vel_world = 0.7 * self.aruco_vel_world + 0.3 * vel
                            
                            vel_msg = TwistStamped()
                            vel_msg.header = pose_msg.header
                            vel_msg.twist.linear.x = self.aruco_vel_world[0]
                            vel_msg.twist.linear.y = self.aruco_vel_world[1]
                            vel_msg.twist.linear.z = self.aruco_vel_world[2]
                            self.platform_vel_pub.publish(vel_msg)

                    self.last_aruco_pos_world = pos_world_marker
                    self.last_aruco_time = current_time

                    # Draw ArUco
                    cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.3)
                else:
                    # This branch should not be reachable now due to fallback matrix
                    cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    cv2.putText(cv_image, "WAITING FOR CAMERA_INFO", (10, height - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                
            # Publish Compressed Debug Image
            debug_msg = CompressedImage()
            debug_msg.header = msg.header
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            self.debug_image_pub.publish(debug_msg)

        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
