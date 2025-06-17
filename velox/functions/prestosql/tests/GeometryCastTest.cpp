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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions::prestosql {

class GeometryCastTest : public functions::test::FunctionBaseTest {
 protected:
  std::optional<std::string> castFromVarbinary(
    const std::optional<std::string>& input) {
    auto result = evaluateOnce<std::string>(
      "cast(cast(from_hex(c0) as geometry) as varchar)", input);
    return result;
  }

  std::optional<std::string> castToVarchar(
      const std::optional<std::string>& input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(c0 as geometry) as varchar)", input);
    return result;
  }
};

TEST_F(GeometryCastTest, invalidWkt) {
  VELOX_ASSERT_USER_THROW(
      castToVarchar(""), "Expected word but encountered end of stream");
  VELOX_ASSERT_USER_THROW(
      castToVarchar("RANDOM_TEXT"), "Unknown type: 'RANDOM_TEXT'");
  VELOX_ASSERT_USER_THROW(
      castToVarchar("LINESTRING (1 1)"),
      "point array must contain 0 or >1 elements");
  VELOX_ASSERT_USER_THROW(
      castToVarchar("LINESTRING ()"), "Expected number but encountered ')'");
  VELOX_ASSERT_USER_THROW(
      castToVarchar("POLYGON ((0 0, 0 0, 0 0, 0 0))"),
      "shell is empty but holes are not");
  VELOX_ASSERT_USER_THROW(
      castToVarchar("POLYGON ((0 0, 0 0))"),
      "Invalid number of points in LinearRing found 2 - must be 0 or >= 3");
  VELOX_ASSERT_USER_THROW(
      castToVarchar("POLYGON ((0 0, 0 1, 1 1, 1 0))"),
      "Points of LinearRing do not form a closed linestring");
}

TEST_F(GeometryCastTest, castEmptyGeometry) {
  EXPECT_EQ(castToVarchar("POINT EMPTY"), "POINT EMPTY");
  EXPECT_EQ(castToVarchar("LINESTRING EMPTY"), "LINESTRING EMPTY");
  EXPECT_EQ(castToVarchar("POLYGON EMPTY"), "POLYGON EMPTY");
  EXPECT_EQ(castToVarchar("MULTIPOINT EMPTY"), "MULTIPOINT EMPTY");
  EXPECT_EQ(castToVarchar("MULTILINESTRING EMPTY"), "MULTILINESTRING EMPTY");
  EXPECT_EQ(castToVarchar("MULTIPOLYGON EMPTY"), "MULTIPOLYGON EMPTY");
  EXPECT_EQ(
      castToVarchar("GEOMETRYCOLLECTION EMPTY"), "GEOMETRYCOLLECTION EMPTY");
}

TEST_F(GeometryCastTest, castToVarchar) {
  EXPECT_EQ(castToVarchar("POINT (1 2)"), "POINT (1 2)");
  EXPECT_EQ(
      castToVarchar("LINESTRING (0 0, 10 10)"), "LINESTRING (0 0, 10 10)");
  EXPECT_EQ(
      castToVarchar("POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0))"),
      "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0))");
  EXPECT_EQ(
      castToVarchar(
          "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))"),
      "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))");
  EXPECT_EQ(
      castToVarchar("MULTIPOINT ((1 2), (3 4))"), "MULTIPOINT ((1 2), (3 4))");
  EXPECT_EQ(
      castToVarchar("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"),
      "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))");
  EXPECT_EQ(
      castToVarchar(
          "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))"),
      "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))");
  EXPECT_EQ(
      castToVarchar("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))"),
      "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))");
}

TEST_F(GeometryCastTest, castToVarbinary) {
  // SRID: 0 POINT (1 2)
  EXPECT_EQ(castFromVarbinary("000000000001000000000000000000F03F0000000000000040"), "POINT (1 2)");
  // SRID: 4326 POLYLINE (0 0, 10 10)
  EXPECT_EQ(castFromVarbinary("E6100000020300000000000000000000000000000000000000000000000000244000000000000024400100000002000000000000000000000000000000000000000000000000000000000024400000000000002440"), "LINESTRING (0 0, 10 10)");
  // SRID: 4326 POLYLINE ((0 0, 10 10), (0 10, 10 0))
  EXPECT_EQ(castFromVarbinary("E6100000020300000000000000000000000000000000000000000000000000244000000000000024400200000004000000000000000200000000000000000000000000000000000000000000000000244000000000000024400000000000000000000000000000244000000000000024400000000000000000"), "MULTILINESTRING ((0 0, 10 10), (0 10, 10 0))");
  // SRID: 3857 POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0))
  EXPECT_EQ(castFromVarbinary("110F0000040500000000000000000000000000000000000000000000000000144000000000000014400100000005000000000000000000000000000000000000000000000000000000000000000000000000001440000000000000144000000000000014400000000000001440000000000000000000000000000000000000000000000000"), "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0))");
  // SRID: 4979 POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))
  EXPECT_EQ(castFromVarbinary("7313000004050000000000000000000000000000000000000000000000000014400000000000001440020000000A00000000000000050000000000000000000000000000000000000000000000000000000000000000001440000000000000144000000000000014400000000000001440000000000000000000000000000000000000000000000000000000000000F03F000000000000F03F0000000000001040000000000000F03F00000000000010400000000000001040000000000000F03F0000000000001040000000000000F03F000000000000F03F"), "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))");
  // SRID: 4979 POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 1 4, 4 4, 4 1, 1 1))
  EXPECT_EQ(castFromVarbinary("7313000005050000000000000000000000000000000000000000000000000014400000000000001440020000000A00000000000000050000000000000000000000000000000000000000000000000000000000000000001440000000000000144000000000000014400000000000001440000000000000000000000000000000000000000000000000000000000000F03F000000000000F03F000000000000F03F0000000000001040000000000000104000000000000010400000000000001040000000000000F03F000000000000F03F000000000000F03F"), "MULTIPOLYGON (((0 0, 0 5, 5 5, 5 0, 0 0)), ((1 1, 1 4, 4 4, 4 1, 1 1)))");
  // SRID: 32633 MULTIPOINT ((1 2), (1 2))")
  EXPECT_EQ(castFromVarbinary("000000000108000000000000000000F03F0000000000000040000000000000F03F000000000000004002000000000000000000F03F0000000000000040000000000000F03F0000000000000040"), "MULTIPOINT ((1 2), (1 2))");
}
} // namespace facebook::velox::functions::prestosql
