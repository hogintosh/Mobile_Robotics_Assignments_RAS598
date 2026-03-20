from setuptools import setup
from glob import glob
import os

package_name = 'perception_assignment'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sean Vellequette',
    maintainer_email='svellequ@asu.edu',
    description='Assignment 1 perception pipeline',
    license='MIT',
    entry_points={
        'console_scripts': [
            'cylinder_processor = perception_assignment.cylinder_processor:main',
        ],
    },
)