###############################################################################
#
# Copyright (C) 2019 Andrew Muzikin
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
    name='tradeflow',
    description='modelling algorithmic trading domains as directed acyclic graphs',
    keywords='',
    author='Andrew Muzikin',
    author_email='muzikinae@gmail.com',
    url='https://github.com/Kismuz/tradeflow',
    project_urls={

        'Source': 'https://github.com/Kismuz/tradeflow',

    },
    license='GPLv3+',
    classifiers=[
        'Development Status :: 1 - Planning',
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
        'gym',
        'ray',
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