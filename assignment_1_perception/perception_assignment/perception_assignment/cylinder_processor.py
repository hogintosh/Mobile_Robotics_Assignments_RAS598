import struct
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


# =========================================================
# CONFIG
# =========================================================
class PipelineConfig:
    def __init__(self):
        # Input topic from the assignment bags
        self.topic = '/oakd/points'

        # Workspace filter
        self.box_min = np.array([-1.0, -0.6, 0.2], dtype=np.float32)
        self.box_max = np.array([ 1.0,  0.6, 2.0], dtype=np.float32)

        # Downsampling
        self.voxel_size = 0.02

        # Normals
        self.k_neighbors = 15

        # Plane RANSAC
        self.floor_dist = 0.02
        self.target_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Y-up
        self.normal_thresh = 0.85
        self.plane_iters = 150

        # Cylinder RANSAC
        self.cyl_radius = 0.055
        self.cyl_radius_tol = 0.02
        self.cylinder_height = 0.40
        self.cylinder_iters = 300

        # Simple clustering
        self.cluster_dist = 0.08
        self.min_cluster_size = 40
        self.max_cluster_size = 4000

        # Safety
        self.max_points_after_downsample = 12000


# =========================================================
# POINT CLOUD HELPERS
# =========================================================
def pointcloud2_to_xyzrgb(msg: PointCloud2):
    """
    Convert PointCloud2 -> xyz, rgb arrays.
    Assumes fields x,y,z and rgb or rgba exist.
    """
    field_names = [f.name for f in msg.fields]
    offsets = {f.name: f.offset for f in msg.fields}

    if not all(name in field_names for name in ['x', 'y', 'z']):
        raise ValueError("PointCloud2 missing x/y/z fields")

    rgb_field = None
    if 'rgb' in field_names:
        rgb_field = 'rgb'
    elif 'rgba' in field_names:
        rgb_field = 'rgba'

    step = msg.point_step
    npts = msg.width * msg.height
    data = msg.data

    xyz = np.zeros((npts, 3), dtype=np.float32)
    rgb = np.zeros((npts, 3), dtype=np.float32)

    ox = offsets['x']
    oy = offsets['y']
    oz = offsets['z']
    orgb = offsets[rgb_field] if rgb_field is not None else None

    for i in range(npts):
        base = i * step
        x = struct.unpack_from('f', data, base + ox)[0]
        y = struct.unpack_from('f', data, base + oy)[0]
        z = struct.unpack_from('f', data, base + oz)[0]
        xyz[i] = [x, y, z]

        if orgb is not None:
            packed = struct.unpack_from('I', data, base + orgb)[0]
            r = (packed >> 16) & 0xFF
            g = (packed >> 8) & 0xFF
            b = packed & 0xFF
            rgb[i] = [r / 255.0, g / 255.0, b / 255.0]

    mask = np.isfinite(xyz).all(axis=1) # Nan filtering to mitigate operations from breaking
    return xyz[mask], rgb[mask]


def xyzrgb_to_pointcloud2(points, colors, frame_id): # Converts back to PointCloud2 for debug publishing in Rviz 
    """
    Create PointCloud2 from Nx3 xyz and Nx3 rgb floats in [0,1].
    """
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(len(points))
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]

    buf = bytearray()
    for p, c in zip(points, colors):
        r = int(np.clip(c[0], 0.0, 1.0) * 255.0)
        g = int(np.clip(c[1], 0.0, 1.0) * 255.0)
        b = int(np.clip(c[2], 0.0, 1.0) * 255.0)
        packed = (r << 16) | (g << 8) | b
        buf.extend(struct.pack('fffI', float(p[0]), float(p[1]), float(p[2]), packed))

    msg.data = bytes(buf)
    return msg


# =========================================================
# RVIZ MARKERS
# =========================================================
class CylinderVisualizer:
    def __init__(self, publisher):
        self.pub_markers = publisher

    def create_cylinder_marker(self, center, radius, rgb, marker_id, frame_id, name='cylinder'):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp.sec = 0
        m.header.stamp.nanosec = 0
        m.ns = name
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        # Assignment skeleton assumes cylinders upright and snaps to floor-like placement
        m.pose.position.x = float(center[0])
        m.pose.position.y = float(0.0)
        m.pose.position.z = float(center[2])

        # Rotate default RViz cylinder axis
        m.pose.orientation.x = 0.7071
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 0.7071

        m.scale.x = float(2.0 * radius)
        m.scale.y = float(2.0 * radius)
        m.scale.z = 0.40

        m.color.r = float(rgb[0])
        m.color.g = float(rgb[1])
        m.color.b = float(rgb[2])
        m.color.a = 0.85
        return m

    def publish_viz(self, cylinders, frame_id):
        ma = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        ma.markers.append(clear_marker)

        for i, cyl in enumerate(cylinders):
            center = cyl['center']
            radius = cyl['radius']
            rgb = cyl['rgb']
            label = cyl['label']
            ma.markers.append(
                self.create_cylinder_marker(center, radius, rgb, 2000 + i, frame_id, label)
            )

        self.pub_markers.publish(ma)


# =========================================================
# PIPELINE
# =========================================================
class CylinderPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def box_filter(self, pts, colors): # keeps points within a nearby location 
        mask = np.all((pts >= self.cfg.box_min) & (pts <= self.cfg.box_max), axis=1)
        return pts[mask], colors[mask]

    def downsample(self, pts, colors): # downsamples by voxel downsampling
        if len(pts) == 0:
            return pts, colors

        vox = np.floor(pts / self.cfg.voxel_size).astype(np.int32) # maps each point to an integer voxel coordinate, partitioning space into cubes and assigning each point to its containing voxel
        _, unique_idx = np.unique(vox, axis=0, return_index=True) # keeps one point per voxel, reduces point count to gain speed
        unique_idx = np.sort(unique_idx)

        pts_ds = pts[unique_idx] # keeps one point per voxel and saves to variable for point positon and color 
        colors_ds = colors[unique_idx]

        if len(pts_ds) > self.cfg.max_points_after_downsample: # limits runtime worst case scenarios by randomly subsampling if still too many points after voxel downsampling
            keep = np.random.choice(len(pts_ds), self.cfg.max_points_after_downsample, replace=False)
            pts_ds = pts_ds[keep]
            colors_ds = colors_ds[keep]

        return pts_ds, colors_ds

    def estimate_normals(self, pts, k=None): # computes normals for each point using a local SVD 
        if k is None:
            k = self.cfg.k_neighbors

        n = len(pts)
        if n < max(k, 3):
            return np.zeros((n, 3), dtype=np.float32)

        tree = cKDTree(pts) # for each point find its k nearest neighbors, we use this because we need multiple close points to estimate a plane and its surface normal
        _, idxs = tree.query(pts, k=k)

        normals = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            nbrs = pts[idxs[i]]
            mu = np.mean(nbrs, axis=0) # Shifts the neighbors so its centriod is at the origin, SVD is based on the variance around the mean, i mean duh doi.
            centered = nbrs - mu

            # SVD: smallest singular vector is local normal
            _, _, vh = np.linalg.svd(centered, full_matrices=False) # Least squares plane fit, the normal is the direction with least variance, which is the last singular vector
            nrm = vh[-1]

            # orient somewhat consistently
            if np.dot(nrm, self.cfg.target_normal) < 0: # What we covered in class, The normal vector can be positive or negative when we compute, so we flip them in one positive direction
                nrm = -nrm

            normals[i] = nrm.astype(np.float32)

        return normals

    def find_plane_ransac(self, pts, iters=None): # Task 1: Detects floor plane using RANSAC -Random Smaple -fit a model -count inliers - repear until we have a model with the most inliers
        if iters is None:
            iters = self.cfg.plane_iters

        npts = len(pts)
        if npts < 3:
            return None, np.array([], dtype=np.int32)

        best_inliers = np.array([], dtype=np.int32)
        best_model = None

        for _ in range(iters):
            ids = np.random.choice(npts, 3, replace=False)
            p1, p2, p3 = pts[ids] # The cross product of 3 non-collinear points gives us the normal of the plane they define 
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm

            align = abs(np.dot(normal, self.cfg.target_normal)) #reject planes whose normals are not close enough to the expected floor normal. 
            if align < self.cfg.normal_thresh:
                continue

            d = -np.dot(normal, p1) # given a unit normal and a point p1, the plane equation is n * x + d = 0, so d = -n * p1
            dist = np.abs(pts @ normal + d) # How inliers are defined with perpendicular distance to the plane.
            inliers = np.where(dist < self.cfg.floor_dist)[0] # Takes the plane with the largest inlier set

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (normal, d)

        return best_model, best_inliers

    def remove_indices(self, pts, colors, normals, remove_idx): # Removes floor inliers from points, colors, normals. When the floor is found it must be removed from all arrays to prevent misalignment
        if len(remove_idx) == 0:
            return pts, colors, normals
        keep = np.ones(len(pts), dtype=bool)
        keep[remove_idx] = False
        return pts[keep], colors[keep], normals[keep]

    def euclidean_clusters(self, pts): # Task 2/3: performs euclidean clustering using BFS over KDTree neighborhood, because remaining points probably belong to cylinders
        """
        Simple BFS clustering using KDTree radius queries.
        """
        n = len(pts)
        if n == 0:
            return []

        tree = cKDTree(pts)
        visited = np.zeros(n, dtype=bool)
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            queue = [i]
            visited[i] = True
            cluster = []

            while queue:
                cur = queue.pop()
                cluster.append(cur)

                nbrs = tree.query_ball_point(pts[cur], self.cfg.cluster_dist) # finds all neighbors within a radius, faster than checking point pairs
                for nb in nbrs:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            if self.cfg.min_cluster_size <= len(cluster) <= self.cfg.max_cluster_size: # removes noise clusters that are too small or too large to be cylinders
                clusters.append(np.array(cluster, dtype=np.int32))

        return clusters

    def rgb_to_hsv_single(self, r, g, b): # Task 4: converts RGB to HSV for simple color classification, we only need hue and saturation for basic color classification for cylinders used in visualization
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if df == 0:
            h = 0.0
        elif mx == r:
            h = (60.0 * ((g - b) / df) + 360.0) % 360.0
        elif mx == g:
            h = (60.0 * ((b - r) / df) + 120.0) % 360.0
        else:
            h = (60.0 * ((r - g) / df) + 240.0) % 360.0

        s = 0.0 if mx == 0 else df / mx
        v = mx
        return h, s, v

    def classify_color(self, colors): # estimates color of cylinder for semantic labeling
        if len(colors) == 0:
            return 'unknown', np.array([1.0, 1.0, 1.0], dtype=np.float32)

        mean_rgb = np.mean(colors, axis=0)
        h, s, v = self.rgb_to_hsv_single(mean_rgb[0], mean_rgb[1], mean_rgb[2])

        # Simple color logic for the course bags
        if s < 0.2 or v < 0.15:
            return 'unknown', mean_rgb

        if 80 <= h <= 160:
            return 'green', np.array([0.1, 0.9, 0.1], dtype=np.float32)
        elif 0 <= h <= 20 or 340 <= h <= 360:
            return 'red', np.array([0.9, 0.1, 0.1], dtype=np.float32)
        elif 200 <= h <= 260:
            return 'blue', np.array([0.1, 0.2, 0.9], dtype=np.float32)
        elif 20 < h < 70:
            return 'yellow', np.array([0.95, 0.9, 0.1], dtype=np.float32)

        return 'unknown', mean_rgb.astype(np.float32)

    def point_to_axis_distances(self, pts, axis_point, axis_dir):
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
        v = pts - axis_point
        proj = np.outer(v @ axis_dir, axis_dir)
        perp = v - proj
        return np.linalg.norm(perp, axis=1)

    def find_single_cylinder(self, pts, normals, colors, iters=None):
        """
        First-pass cylinder RANSAC:
        - sample 2 points and normals
        - axis direction ~ cross(n1, n2)
        - estimate axis point from the midpoint
        - score by how many points have distance near expected radius
        """
        if iters is None:
            iters = self.cfg.cylinder_iters

        n = len(pts)
        if n < 20:
            return None

        best = None
        best_inliers = np.array([], dtype=np.int32)

        for _ in range(iters):
            i, j = np.random.choice(n, 2, replace=False)
            p1, p2 = pts[i], pts[j]
            n1, n2 = normals[i], normals[j]

            axis_dir = np.cross(n1, n2)
            axis_norm = np.linalg.norm(axis_dir)
            if axis_norm < 1e-6:
                continue
            axis_dir = axis_dir / axis_norm

            # For upright-ish cylinders in these bags, prefer axis near Y
            if abs(np.dot(axis_dir, self.cfg.target_normal)) < 0.6:
                continue

            axis_point = 0.5 * (p1 + p2)

            dists = self.point_to_axis_distances(pts, axis_point, axis_dir)
            inliers = np.where(np.abs(dists - self.cfg.cyl_radius) < self.cfg.cyl_radius_tol)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

                center = np.mean(pts[inliers], axis=0) if len(inliers) > 0 else axis_point
                label, viz_rgb = self.classify_color(colors[inliers]) if len(inliers) > 0 else ('unknown', np.array([1, 1, 1], dtype=np.float32))

                best = {
                    'center': center.astype(np.float32),
                    'axis': axis_dir.astype(np.float32),
                    'radius': float(self.cfg.cyl_radius),
                    'inliers': best_inliers,
                    'rgb': viz_rgb.astype(np.float32),
                    'label': label,
                }

        return best

    def run(self, pts, colors):
        debug = {}

        # Stage 0: workspace crop
        pts0, col0 = self.box_filter(pts, colors)
        debug['stage0_box'] = (pts0, col0)

        # Stage 1: downsample
        pts1, col1 = self.downsample(pts0, col0)
        debug['stage1_downsampled'] = (pts1, col1)

        if len(pts1) < 30:
            return [], debug

        # Stage 2: normals
        normals = self.estimate_normals(pts1)

        # Stage 3: floor plane
        _, floor_inliers = self.find_plane_ransac(pts1)
        pts2, col2, nrm2 = self.remove_indices(pts1, col1, normals, floor_inliers)
        debug['stage2_nofloor'] = (pts2, col2)

        if len(pts2) < 20:
            return [], debug

        # Stage 4: clustering
        clusters = self.euclidean_clusters(pts2)

        # Flatten for RViz debug
        if len(clusters) > 0:
            all_idx = np.concatenate(clusters)
            debug['stage3_candidates'] = (pts2[all_idx], col2[all_idx])
        else:
            debug['stage3_candidates'] = (np.empty((0, 3), dtype=np.float32),
                                          np.empty((0, 3), dtype=np.float32))

        cylinders = []
        for cl in clusters[:3]:
            cyl = self.find_single_cylinder(pts2[cl], nrm2[cl], col2[cl])
            if cyl is not None and len(cyl['inliers']) >= self.cfg.min_cluster_size // 2:
                cylinders.append(cyl)

        return cylinders, debug


# =========================================================
# ROS NODE
# =========================================================
class CylinderProcessorNode(Node):
    def __init__(self):
        super().__init__('cylinder_processor')

        self.cfg = PipelineConfig()
        self.pipeline = CylinderPipeline(self.cfg)

        self.sub = self.create_subscription(
            PointCloud2,
            self.cfg.topic,
            self.cloud_callback,
            10
        )

        self.pub_markers = self.create_publisher(MarkerArray, '/viz/detections', 10)
        self.pub_stage0 = self.create_publisher(PointCloud2, '/pipeline/stage0_box', 10)
        self.pub_stage1 = self.create_publisher(PointCloud2, '/pipeline/stage1_downsampled', 10)
        self.pub_stage2 = self.create_publisher(PointCloud2, '/pipeline/stage2_nofloor', 10)
        self.pub_stage3 = self.create_publisher(PointCloud2, '/pipeline/stage3_candidates_pc', 10)

        self.visualizer = CylinderVisualizer(self.pub_markers)

        self.get_logger().info('Cylinder processor node started.')
        self.get_logger().info(f'Subscribed to {self.cfg.topic}')

    def publish_debug_cloud(self, pub, frame_id, pts, colors):
        msg = xyzrgb_to_pointcloud2(pts, colors, frame_id)
        pub.publish(msg)

    def cloud_callback(self, msg: PointCloud2):
        try:
            pts, colors = pointcloud2_to_xyzrgb(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to decode PointCloud2: {e}')
            return

        cylinders, debug = self.pipeline.run(pts, colors)

        frame_id = msg.header.frame_id

        if 'stage0_box' in debug:
            self.publish_debug_cloud(self.pub_stage0, frame_id, *debug['stage0_box'])
        if 'stage1_downsampled' in debug:
            self.publish_debug_cloud(self.pub_stage1, frame_id, *debug['stage1_downsampled'])
        if 'stage2_nofloor' in debug:
            self.publish_debug_cloud(self.pub_stage2, frame_id, *debug['stage2_nofloor'])
        if 'stage3_candidates' in debug:
            self.publish_debug_cloud(self.pub_stage3, frame_id, *debug['stage3_candidates'])

        self.visualizer.publish_viz(cylinders, frame_id)

        self.get_logger().info(
            f'raw={len(pts)} '
            f'box={len(debug["stage0_box"][0]) if "stage0_box" in debug else 0} '
            f'ds={len(debug["stage1_downsampled"][0]) if "stage1_downsampled" in debug else 0} '
            f'nofloor={len(debug["stage2_nofloor"][0]) if "stage2_nofloor" in debug else 0} '
            f'cylinders={len(cylinders)}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CylinderProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()