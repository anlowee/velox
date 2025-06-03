# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
include_guard(GLOBAL)
# H3 Configuration
set(VELOX_H3_BUILD_VERSION 4.2.1)
set(VELOX_H3_BUILD_SHA256_CHECKSUM
        1b51822b43f3887767c5a5aafd958fca72b72fc454f3b3f6eeea31757d74687d)
string(CONCAT VELOX_H3_SOURCE_URL "https://github.com/uber/h3/archive/refs/tags/v${VELOX_H3_BUILD_VERSION}.tar.gz")

velox_resolve_dependency_url(H3)

FetchContent_Declare(
        h3
        URL ${VELOX_H3_SOURCE_URL}
        URL_HASH ${VELOX_H3_BUILD_SHA256_CHECKSUM})

list(APPEND CMAKE_MODULE_PATH "${h3_SOURCE_DIR}/cmake")
set(CMAKE_BUILD_TYPE Release)

FetchContent_MakeAvailable(h3)

# Create an alias target so other components can link easily
add_library(H3::h3 ALIAS h3)

# Restore previous build settings
unset(BUILD_SHARED_LIBS)
set(CMAKE_CXX_FLAGS ${PREVIOUS_CMAKE_CXX_FLAGS})
set(CMAKE_BUILD_TYPE ${PREVIOUS_BUILD_TYPE})
