# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = "camtones",
    version = "0.0.1",
    description = "Camera To Not-Empty Sequence is a motion and face extration system",
    author = "Jesús Espino",
    author_email = "jespinog@gmail.com",
    url = "https://github.com/jespino/camtones",
    scripts = ['scripts/camtones'],
    packages = find_packages(),
    package_data={
        'camtones': [
            'haars/*',
            'procs/gui.glade',
        ]
    },
    install_requires=[
        'imutils',
        'numpy',
        'click',
    ],
    license = "BSD",
    classifiers = [
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.4',
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
    ],
)
