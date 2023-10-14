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
Email: rameez.ismaeel@gmail.com
"""

import os
import subprocess
from pathlib import Path


def extract_frames(parent_dir: os.PathLike, category: str = 'side_angles'):
    for dish in list(Path(parent_dir, 'imagery', category).rglob('*.h264'))[0:1]:
        frames_dir = f'{str(dish)[:-5]}.frames'
        os.mkdir(frames_dir) if not os.path.exists(frames_dir) else None
        
        cmd = f"ffmpeg -i {str(dish)} -filter:v \"select=not(mod(n\,4))\" " \
              f"{str(Path(frames_dir, '%03d.jpeg'))}" \
              f"  -vsync 0"
        try:
            print(f'Executing: {cmd}')
            subproc = subprocess.run(cmd, capture_output=True)
            assert subproc.returncode == 0
        except AssertionError:
            raise AssertionError('Subprocess Failure\n:'
                                 'The subprocess invoked by the following command failed\n'
                                 f'{cmd}')


if __name__ == '__main__':
    extracted = Path(os.path.dirname(__file__), 'dummy_data', 'nutrition5k_dataset')
    extract_frames(extracted)