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

# velox.py.type library:
velox_add_library(velox_py_type_lib type/PyType.cpp)
velox_link_libraries(velox_py_type_lib velox_type pybind11::module)

pybind11_add_module(type MODULE type/type.cpp)
target_link_libraries(
  type
  PRIVATE velox_py_type_lib)

# velox.py.file library:
velox_add_library(velox_py_file_lib file/PyFile.cpp)
velox_link_libraries(velox_py_file_lib velox_dwio_common pybind11::module)

pybind11_add_module(file MODULE file/file.cpp)
target_link_libraries(
  file
  PRIVATE velox_py_file_lib)

# velox.py.vector library:
velox_add_library(velox_py_vector_lib vector/PyVector.cpp)
velox_link_libraries(velox_py_vector_lib velox_vector pybind11::module)

pybind11_add_module(vector MODULE vector/vector.cpp)
target_link_libraries(
  vector
  PRIVATE velox_py_vector_lib)

# velox.py.plan_builder library:
velox_add_library(velox_py_plan_builder_lib plan_builder/PyPlanBuilder.cpp)
velox_link_libraries(
  velox_py_plan_builder_lib
  velox_py_type_lib
  velox_py_vector_lib
  velox_vector
  velox_core
  velox_cursor
  velox_hive_connector
  velox_aggregates
  velox_functions_prestosql
  velox_parse_expression
  velox_exec_test_lib
  velox_tpch_connector
  velox_dwio_dwrf_reader
  velox_dwio_dwrf_common
  velox_dwio_dwrf_writer
  Folly::folly
  pybind11::module)

pybind11_add_module(plan_builder MODULE plan_builder/plan_builder.cpp)
target_link_libraries(
  plan_builder
  PRIVATE velox_py_plan_builder_lib)

# velox.py.runner library:
velox_add_library(velox_py_local_runner_lib runner/PyLocalRunner.cpp
                  runner/PyConnectors.cpp)
velox_link_libraries(
  velox_py_local_runner_lib
  velox_py_type_lib
  velox_py_vector_lib
  velox_vector
  velox_core
  velox_cursor
  velox_hive_connector
  velox_exec_test_lib
  velox_dwio_dwrf_reader
  velox_dwio_dwrf_common
  velox_dwio_dwrf_writer
  pybind11::module)

pybind11_add_module(runner MODULE runner/runner.cpp)
target_link_libraries(
  runner
  PRIVATE velox_py_local_runner_lib)
