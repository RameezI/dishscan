# Copyright (c) 2023 Rameez Ismail
#
# Licensed under the The MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#    https://opensource.org/license/mit/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Author(s): Rameez Ismail
# Email(s):  rameez.ismail@protonmail.com

"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: Rameez.ismail@philips.com
"""

from setuptools import setup

_MAJOR_VERSION = '0'
_MINOR_VERSION = '1'
_PATCH_VERSION = '0'

__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])

setup(name='dishscan',
      version=__version__,
      description="""
      A set of experiments focused on the automatic understanding of
      the constituents of generic food plates.
      """,
      url='',
      author='Rameez Ismail',
      author_email='Rameez.ismail@philips.com',
      license='@Copyrights [Royal Philips], all rights reserved',
      packages=['dishscan'],
      install_requires=[],
      zip_safe=False)