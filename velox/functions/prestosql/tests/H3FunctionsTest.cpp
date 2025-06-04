/*
* Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <limits>
#include <optional>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using facebook::velox::functions::test::FunctionBaseTest;

class H3FunctionsTest : public FunctionBaseTest {

};

TEST_F(H3FunctionsTest, getHexagonAddr) {
  const auto hexagonAddr = [&](const double lat, const double lng, const int64_t res) {
    return evaluateOnce<std::string>(
        "get_hexagon_addr(c0, c1, c2)", lat, lng, res);
  };

  std::cout << hexagonAddr(40.730610, -73.935242, 0) << std::endl;
}