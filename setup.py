# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the setup file for airdialogue python library."""

import setuptools
with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="airdialogue-essentials",
    version="0.1",
    scripts=["airdialogue/bin/airdialogue"],
    author="Wei Wei",
    author_email="wewei@google.com",
    description="airdialogue package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/javatechy/dokr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["tqdm", "nltk", "flask"]
)
