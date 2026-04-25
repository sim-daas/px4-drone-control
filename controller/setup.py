from setuptools import setup
import os
from glob import glob

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Actuator-level controller package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offboard_heartbeat = controller.offboard_heartbeat:main',
            'motor_test = controller.motor_test:main',
            'integrated_motor_test = controller.integrated_motor_test:main',
            'rate_controller = controller.rate_controller:main',
            'se3_controller = controller.se3_controller:main',
            'drone_controller = controller.drone_controller:main',
            'position_input = controller.position_input:main',
            'live_plotter = controller.live_plotter:main',
            'sac_controller = controller.sac_controller:main',
        ],
    },
)
