from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'stepper_rl_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Stepper motor RL control package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = stepper_rl_control.training_node:main',
        ],
    },
)
