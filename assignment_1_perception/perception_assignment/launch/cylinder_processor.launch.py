from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('perception_assignment')
    default_params = os.path.join(pkg_share, 'params', 'cylinder_filters.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to parameter YAML file for the cylinder processor'
    )

    node = Node(
        package='perception_assignment',
        executable='cylinder_processor',
        name='cylinder_processor',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    return LaunchDescription([
        params_file_arg,
        node,
    ])
