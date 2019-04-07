###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

from setuptools import setup


setup(
    name='tensortrade',
    description='--',
    keywords='--',
    author='Andrew Muzikin',
    author_email='muzikinae@gmail.com',
    url='https://github.com/Kismuz/tensortrade',
    project_urls={

        'Source': 'https://github.com/Kismuz/tensortrade',

    },
    license='GPLv3+',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Application Frameworks',


    ],
    version='0.0.8',
    install_requires=[
        'pythonflow',
        'numpy',
        'pandas',
        'ipython',
        'psutil',
        'logbook'
    ],
    python_requires='>=3.6',
    include_package_data=True,
)