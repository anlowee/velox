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

#ifdef VELOX_ENABLE_GEO
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class GeometryFunctionsTest : public functions::test::FunctionBaseTest {};

TEST_F(GeometryFunctionsTest, wktAndWkb) {
  const auto wktRoundTrip = [&](const std::optional<std::string>& a) {
    return evaluateOnce<std::string>("ST_AsText(ST_GeometryFromText(c0))", a);
  };

  const auto wktToWkb = [&](const std::optional<std::string>& wkt) {
    return evaluateOnce<std::string>(
        "to_hex(ST_AsBinary(ST_GeometryFromText(c0)))", wkt);
  };

  const auto wkbToWkT = [&](const std::optional<std::string>& wkb) {
    return evaluateOnce<std::string>(
        "ST_AsText(ST_GeomFromBinary(from_hex(c0)))", wkb);
  };

  const auto wkbRoundTrip = [&](const std::optional<std::string>& wkt) {
    return evaluateOnce<std::string>(
        "to_hex(ST_AsBinary(ST_GeomFromBinary(from_hex(c0))))", wkt);
  };

  const std::vector<std::string> wkts = {
      "POINT (1 2)",
      "LINESTRING (0 0, 10 10)",
      "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0))",
      "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (1 1, 4 1, 4 4, 1 4, 1 1))",
      "MULTIPOINT ((1 2), (3 4))",
      "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
      "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))",
      "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))"};

  const std::vector<std::string> littleEndianWkbs = {
      "0101000000000000000000F03F0000000000000040",
      "0102000000020000000000000000000000000000000000000000000000000024400000000000002440",
      "010300000001000000050000000000000000000000000000000000000000000000000000000000000000001440000000000000144000000000000014400000000000001440000000000000000000000000000000000000000000000000",
      "01030000000200000005000000000000000000000000000000000000000000000000000000000000000000144000000000000014400000000000001440000000000000144000000000000000000000000000000000000000000000000005000000000000000000F03F000000000000F03F0000000000001040000000000000F03F00000000000010400000000000001040000000000000F03F0000000000001040000000000000F03F000000000000F03F",
      "0104000000020000000101000000000000000000F03F0000000000000040010100000000000000000008400000000000001040",
      "01050000000200000001020000000200000000000000000000000000000000000000000000000000F03F000000000000F03F0102000000020000000000000000000040000000000000004000000000000008400000000000000840",
      "01060000000200000001030000000100000005000000000000000000000000000000000000000000000000000000000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000000000000000000000000000000000000000010300000001000000050000000000000000000040000000000000004000000000000000400000000000000840000000000000084000000000000008400000000000000840000000000000004000000000000000400000000000000040",
      "0107000000020000000101000000000000000000F03F00000000000000400102000000020000000000000000000840000000000000104000000000000014400000000000001840"};

  const std::vector<std::string> bigEndianWkbs = {
      "00000000013FF00000000000004000000000000000",
      "0000000002000000020000000000000000000000000000000040240000000000004024000000000000",
      "000000000300000001000000050000000000000000000000000000000000000000000000004014000000000000401400000000000040140000000000004014000000000000000000000000000000000000000000000000000000000000",
      "000000000300000002000000050000000000000000000000000000000000000000000000004014000000000000401400000000000040140000000000004014000000000000000000000000000000000000000000000000000000000000000000053ff00000000000003ff000000000000040100000000000003ff0000000000000401000000000000040100000000000003ff000000000000040100000000000003ff00000000000003ff0000000000000",
      "00000000040000000200000000013ff00000000000004000000000000000000000000140080000000000004010000000000000",
      "000000000500000002000000000200000002000000000000000000000000000000003ff00000000000003ff00000000000000000000002000000024000000000000000400000000000000040080000000000004008000000000000",
      "000000000600000002000000000300000001000000050000000000000000000000000000000000000000000000003ff00000000000003ff00000000000003ff00000000000003ff0000000000000000000000000000000000000000000000000000000000000000000000300000001000000054000000000000000400000000000000040000000000000004008000000000000400800000000000040080000000000004008000000000000400000000000000040000000000000004000000000000000",
      "00000000070000000200000000013ff000000000000040000000000000000000000002000000024008000000000000401000000000000040140000000000004018000000000000",
  };

  for (size_t i = 0; i < wkts.size(); i++) {
    EXPECT_EQ(wkts[i], wktRoundTrip(wkts[i]));
    EXPECT_EQ(littleEndianWkbs[i], wktToWkb(wkts[i]));
    EXPECT_EQ(wkts[i], wkbToWkT(littleEndianWkbs[i]));
    EXPECT_EQ(littleEndianWkbs[i], wkbRoundTrip(littleEndianWkbs[i]));

    EXPECT_EQ(littleEndianWkbs[i], wkbRoundTrip(bigEndianWkbs[i]));
    EXPECT_EQ(wkts[i], wkbToWkT(bigEndianWkbs[i]));
  }

  const std::vector<std::string> emptyGeometryWkts = {
      "POINT EMPTY",
      "LINESTRING EMPTY",
      "POLYGON EMPTY",
      "MULTIPOINT EMPTY",
      "MULTILINESTRING EMPTY",
      "MULTIPOLYGON EMPTY",
      "GEOMETRYCOLLECTION EMPTY"};

  const std::vector<std::string> emptyGeometryWkbs = {
      "0101000000000000000000F87F000000000000F87F",
      "010200000000000000",
      "010300000000000000",
      "010400000000000000",
      "010500000000000000",
      "010600000000000000",
      "010700000000000000"};

  for (size_t i = 0; i < emptyGeometryWkts.size(); i++) {
    EXPECT_EQ(wktRoundTrip(emptyGeometryWkts[i]), emptyGeometryWkts[i]);
    EXPECT_EQ(emptyGeometryWkbs[i], wktToWkb(emptyGeometryWkts[i]));
    EXPECT_EQ(emptyGeometryWkts[i], wkbToWkT(emptyGeometryWkbs[i]));
    EXPECT_EQ(emptyGeometryWkbs[i], wkbRoundTrip(emptyGeometryWkbs[i]));
  }

  // WKT invalid cases
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip(""), "Expected word but encountered end of stream");
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip("RANDOM_TEXT"), "Unknown type: 'RANDOM_TEXT'");
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip("LINESTRING (1 1)"),
      "point array must contain 0 or >1 elements");
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip("LINESTRING ()"), "Expected number but encountered ')'");
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip("POLYGON ((0 0, 0 0, 0 0, 0 0))"),
      "shell is empty but holes are not");
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip("POLYGON ((0 0, 0 0))"),
      "Invalid number of points in LinearRing found 2 - must be 0 or >= 3");
  VELOX_ASSERT_USER_THROW(
      wktRoundTrip("POLYGON ((0 0, 0 1, 1 1, 1 0))"),
      "Points of LinearRing do not form a closed linestring");

  // WKB invalid cases
  // Empty
  VELOX_ASSERT_USER_THROW(wkbRoundTrip(""), "Unexpected EOF parsing WKB");

  // Random bytes
  VELOX_ASSERT_USER_THROW(wkbRoundTrip("ABCDEF"), "Unexpected EOF parsing WKB");

  // Unrecognized geometry type
  VELOX_ASSERT_USER_THROW(
      wkbRoundTrip("0109000000000000000000F03F0000000000000040"),
      "Unrecognized geometry type: 9");

  // Point with missing y
  VELOX_ASSERT_USER_THROW(
      wkbRoundTrip("0101000000000000000000F03F"),
      "Input buffer is smaller than requested object size");

  // LineString with only one point
  VELOX_ASSERT_THROW(
      wkbRoundTrip("010200000001000000000000000000F03F000000000000F03F"),
      "point array must contain 0 or >1 elements");

  // Polygon with unclosed LinString
  VELOX_ASSERT_THROW(
      wkbRoundTrip(
          "01030000000100000004000000000000000000000000000000000000000000000000000000000000000000F03F000000000000F03F000000000000F03F000000000000F03F0000000000000000"),
      "Points of LinearRing do not form a closed linestring");

  VELOX_ASSERT_THROW(
      wkbRoundTrip(
          "010300000001000000020000000000000000000000000000000000000000000000000000000000000000000000"),
      "Invalid number of points in LinearRing found 2 - must be 0 or >= 3");

  VELOX_ASSERT_THROW(
      wkbRoundTrip(
          "0103000000010000000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
      "shell is empty but holes are not");
}

TEST_F(GeometryFunctionsTest, stPoint) {
  const auto stPoint = [&](const std::optional<double>& a,
                           const std::optional<double>& b) {
    return evaluateOnce<std::string>("ST_AsText(ST_Point(c0, c1))", a, b);
  };
  EXPECT_EQ(stPoint(3, 5), "POINT (3 5)");
  EXPECT_EQ(stPoint(0, 0), "POINT (0 0)");
  EXPECT_EQ(stPoint(-3, -7), "POINT (-3 -7)");
  EXPECT_EQ(stPoint(1.5, 2.25), "POINT (1.5 2.25)");
  EXPECT_EQ(stPoint(-1000.123, 5000.987), "POINT (-1000.123 5000.987)");
  EXPECT_EQ(stPoint(123456.789, -987654.321), "POINT (123456.789 -987654.321)");

  EXPECT_EQ(stPoint(1, std::nullopt), std::nullopt);
  EXPECT_EQ(stPoint(std::nullopt, 1), std::nullopt);
  EXPECT_EQ(stPoint(std::nullopt, std::nullopt), std::nullopt);
}

TEST_F(GeometryFunctionsTest, stContains) {
  const auto stContains = [&](const std::optional<std::string>& a,
                              const std::optional<std::string>& b) {
    return evaluateOnce<bool>(
        "ST_Contains(ST_GeometryFromText(c0), ST_GeometryFromText(c1))", a, b);
  };

  // Point inside polygon
  EXPECT_EQ(
      stContains("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))", "POINT(5 5)"), true);

  // Point on polygon edge
  EXPECT_EQ(
      stContains("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))", "POINT(0 5)"),
      false);

  // LineString fully inside polygon
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))", "LINESTRING(2 2, 8 8)"),
      true);

  // LineString partially outside polygon
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))", "LINESTRING(2 2, 12 12)"),
      false);

  // Polygon fully inside another polygon
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))",
          "POLYGON((2 2, 2 8, 8 8, 8 2, 2 2))"),
      true);

  // Polygon touching boundary
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))",
          "POLYGON((10 0, 10 2, 12 2, 12 0, 10 0))"),
      false);

  // Point in multipolygon
  EXPECT_EQ(
      stContains(
          "MULTIPOLYGON(((0 0, 0 10, 10 10, 10 0, 0 0)), ((20 20, 20 30, 30 30, 30 20, 20 20)))",
          "POINT(25 25)"),
      true);

  // GeometryCollection
  EXPECT_EQ(
      stContains(
          "GEOMETRYCOLLECTION(POINT(1 1), LINESTRING(0 0, 2 2))", "POINT(1 1)"),
      true);

  // MultiPoint fully inside
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))",
          "MULTIPOINT((1 1), (2 2), (3 3))"),
      true);

  // MultiPoint partially outside
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))",
          "MULTIPOINT((1 1), (2 2), (6 6))"),
      false);

  // Polygon with hole (point in hole)
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
          "POINT(5 5)"),
      false);

  // Polygon with hole (point in outer ring)
  EXPECT_EQ(
      stContains(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
          "POINT(1 1)"),
      true);

  // Point on polygon vertex
  EXPECT_EQ(
      stContains("POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"), false);

  // Large coordinates
  EXPECT_EQ(
      stContains(
          "POLYGON((1000000 1000000, 1000000 2000000, 2000000 2000000, 2000000 1000000, 1000000 1000000))",
          "POINT(1500000 1500000)"),
      true);
}

TEST_F(GeometryFunctionsTest, stWithin) {
  const auto stWithin = [&](const std::optional<std::string>& a,
                            const std::optional<std::string>& b) {
    return evaluateOnce<bool>(
        "ST_Within(ST_GeometryFromText(c0), ST_GeometryFromText(c1))", a, b);
  };

  // Point inside polygon
  EXPECT_EQ(stWithin("POINT(1 1)", "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))"), true);

  // Point on boundary
  EXPECT_EQ(
      stWithin("POINT(0 1)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"), false);

  // Line inside polygon
  EXPECT_EQ(
      stWithin("LINESTRING(1 1, 2 2)", "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))"),
      true);

  // Line crossing polygon boundary
  EXPECT_EQ(
      stWithin("LINESTRING(1 1, 6 6)", "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))"),
      false);

  // Polygon inside polygon
  EXPECT_EQ(
      stWithin(
          "POLYGON((1 1, 1 2, 2 2, 2 1, 1 1))",
          "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))"),
      true);

  // Polygon touches outer polygon
  EXPECT_EQ(
      stWithin(
          "POLYGON((2 2, 2 4, 4 4, 4 2, 2 2))",
          "POLYGON((0 0, 0 4, 2 4, 2 0, 0 0))"),
      false);

  // Point in MultiPolygon
  EXPECT_EQ(
      stWithin(
          "POINT(6 6)",
          "MULTIPOLYGON(((0 0, 0 2, 2 2, 2 0, 0 0)), ((5 5, 5 7, 7 7, 7 5, 5 5)))"),
      true);

  // Point not in MultiPolygon
  EXPECT_EQ(
      stWithin(
          "POINT(10 10)",
          "MULTIPOLYGON(((0 0, 0 2, 2 2, 2 0, 0 0)), ((5 5, 5 7, 7 7, 7 5, 5 5)))"),
      false);

  // Polygon with hole - point in hole
  EXPECT_EQ(
      stWithin(
          "POINT(4 4)",
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (3 3, 3 5, 5 5, 5 3, 3 3))"),
      false);

  // Polygon with hole - point in outer ring
  EXPECT_EQ(
      stWithin(
          "POINT(2 2)",
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (3 3, 3 5, 5 5, 5 3, 3 3))"),
      true);

  // MultiPoint fully inside polygon
  EXPECT_EQ(
      stWithin(
          "MULTIPOINT((1 1), (2 2))", "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))"),
      true);

  // MultiPoint partially outside
  EXPECT_EQ(
      stWithin(
          "MULTIPOINT((1 1), (4 4))", "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))"),
      false);

  // Point in GeometryCollection
  EXPECT_EQ(
      stWithin(
          "POINT(1 1)",
          "GEOMETRYCOLLECTION(POLYGON((0 0, 0 2, 2 2, 2 0, 0 0)))"),
      true);

  // Complex nested polygons
  EXPECT_TRUE(stWithin(
      "POLYGON((2 2, 2 3, 3 3, 3 2, 2 2))",
      "POLYGON((1 1, 1 4, 4 4, 4 1, 1 1))"));

  // Geometry equals container
  EXPECT_EQ(
      stWithin(
          "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
          "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))"),
      true);

  // Same point
  EXPECT_EQ(stWithin("POINT(1 1)", "POINT(1 1)"), true);

  // Large coordinates
  EXPECT_EQ(
      stWithin(
          "POINT(1500000 1500000)",
          "POLYGON((1000000 1000000, 1000000 2000000, 2000000 2000000, 2000000 1000000, 1000000 1000000))"),
      true);
}

TEST_F(GeometryFunctionsTest, stDistance) {
  const auto stDistance = [&](const std::optional<std::string>& a,
                              const std::optional<std::string>& b) {
    return evaluateOnce<double>(
        "ST_Distance(ST_GeometryFromText(c0), ST_GeometryFromText(c1))", a, b);
  };

  // Distance between same point
  EXPECT_DOUBLE_EQ(stDistance("POINT(0 0)", "POINT(0 0)").value(), 0.0);

  // Horizontal distance
  EXPECT_DOUBLE_EQ(stDistance("POINT(0 0)", "POINT(3 0)").value(), 3.0);

  // Vertical distance
  EXPECT_DOUBLE_EQ(stDistance("POINT(0 0)", "POINT(0 4)").value(), 4.0);

  // Diagonal distance
  EXPECT_DOUBLE_EQ(stDistance("POINT(0 0)", "POINT(3 4)").value(), 5.0);

  // Point to LineString
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(0 0)", "LINESTRING(1 0, 1 2)").value(), 1.0);

  // Point to Polygon (inside)
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(1 1)", "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))").value(),
      0.0);

  // Point to Polygon (outside)
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(5 5)", "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))").value(),
      std::sqrt(8.0));

  // Polygon with hole — point in hole
  EXPECT_DOUBLE_EQ(
      stDistance(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (3 3, 3 7, 7 7, 7 3, 3 3))",
          "POINT(5 5)")
          .value(),
      2.0);

  // Point to MultiPoint
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(1 1)", "MULTIPOINT((2 2), (3 3))").value(),
      std::sqrt(2.0));

  // Point to MultiLineString
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(0 0)", "MULTILINESTRING((1 0, 1 2), (5 5, 6 6))")
          .value(),
      1.0);

  // LineString to LineString
  EXPECT_DOUBLE_EQ(
      stDistance("LINESTRING(0 0, 1 1)", "LINESTRING(2 2, 3 3)").value(),
      std::sqrt(2.0));

  // Polygon to Polygon (overlapping)
  EXPECT_DOUBLE_EQ(
      stDistance(
          "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))",
          "POLYGON((2 2, 2 5, 5 5, 5 2, 2 2))")
          .value(),
      0.0);

  // Polygon to Polygon (apart)
  EXPECT_DOUBLE_EQ(
      stDistance(
          "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
          "POLYGON((3 3, 3 4, 4 4, 4 3, 3 3))")
          .value(),
      std::sqrt(8.0));

  // Point to GeometryCollection
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(0 0)", "GEOMETRYCOLLECTION(POINT(5 0))").value(), 5.0);

  // Large geometry distance
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(0 0)", "LINESTRING(1000000 1000000, 2000000 2000000)")
          .value(),
      std::sqrt(2.0) * 1000000);

  // Negative coordinate geometries
  EXPECT_DOUBLE_EQ(
      stDistance("POINT(-1 -1)", "POINT(-4 -5)").value(), std::sqrt(25.0));
}

TEST_F(GeometryFunctionsTest, stCentroid) {
  const auto stCentroid = [&](const std::optional<std::string>& a) {
    return evaluateOnce<std::string>(
        "ST_AsText(ST_Centroid(ST_GeometryFromText(c0)))", a);
  };

  // Centroid of a square
  EXPECT_EQ(stCentroid("POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"), "POINT (1 1)");

  // LineString centroid
  EXPECT_EQ(stCentroid("LINESTRING(0 0, 2 0)"), "POINT (1 0)");

  // Point centroid
  EXPECT_EQ(stCentroid("POINT(1 2)"), "POINT (1 2)");

  // MultiPoint centroid
  EXPECT_EQ(stCentroid("MULTIPOINT((1 1), (3 3))"), "POINT (2 2)");

  // MultiLineString centroid
  EXPECT_EQ(
      stCentroid("MULTILINESTRING((0 0, 2 0), (2 0, 2 2))"), "POINT (1.5 0.5)");

  // Polygon with hole (should ignore hole for centroid)
  EXPECT_EQ(
      stCentroid(
          "POLYGON((0 0, 0 6, 6 6, 6 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"),
      "POINT (3 3)");

  // Negative coordinates
  EXPECT_EQ(
      stCentroid("POLYGON((-4 -4, -4 -2, -2 -2, -2 -4, -4 -4))"),
      "POINT (-3 -3)");

  // GeometryCollection
  EXPECT_EQ(
      stCentroid("GEOMETRYCOLLECTION(POINT(0 0), POINT(2 2))"), "POINT (1 1)");

  // Centroid of L-shape polygon
  EXPECT_EQ(
      stCentroid("POLYGON((0 0, 0 3, 1 3, 1 1, 3 1, 3 0, 0 0))"),
      "POINT (1.1 1.1)");

  // Tiny polygon
  EXPECT_EQ(
      stCentroid("POLYGON((0 0, 0 0.001, 0.001 0.001, 0.001 0, 0 0))"),
      "POINT (0.0005 0.0005)");

  // Large polygon
  EXPECT_EQ(
      stCentroid(
          "POLYGON((1000000 1000000, 1000000 2000000, 2000000 2000000, 2000000 1000000, 1000000 1000000))"),
      "POINT (1500000 1500000)");
}

TEST_F(GeometryFunctionsTest, stPolygon) {
  const auto stPolygon = [&](const std::optional<std::string>& wkt) {
    return evaluateOnce<std::string>("ST_AsText(ST_Polygon(c0))", wkt);
  };

  // Simple square
  EXPECT_EQ(
      stPolygon("POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"),
      "POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))");

  // Rectangle
  EXPECT_EQ(
      stPolygon("POLYGON((10 10, 10 20, 30 20, 30 10, 10 10))"),
      "POLYGON ((10 10, 10 20, 30 20, 30 10, 10 10))");

  // Triangle
  EXPECT_EQ(
      stPolygon("POLYGON((0 0, 0 3, 3 0, 0 0))"),
      "POLYGON ((0 0, 0 3, 3 0, 0 0))");

  // Thin rectangle
  EXPECT_EQ(
      stPolygon("POLYGON((0 0, 0 1, 10 1, 10 0, 0 0))"),
      "POLYGON ((0 0, 0 1, 10 1, 10 0, 0 0))");

  // Square with a hole
  EXPECT_EQ(
      stPolygon(
          "POLYGON((0 0, 0 6, 6 6, 6 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"),
      "POLYGON ((0 0, 0 6, 6 6, 6 0, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))");

  // L-shaped polygon
  EXPECT_EQ(
      stPolygon("POLYGON((0 0, 0 4, 2 4, 2 2, 4 2, 4 0, 0 0))"),
      "POLYGON ((0 0, 0 4, 2 4, 2 2, 4 2, 4 0, 0 0))");

  // Hole touches outer ring
  EXPECT_EQ(
      stPolygon(
          "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0), (0 0, 0 1, 1 1, 1 0, 0 0))"),
      "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (0 0, 1 0, 1 1, 0 1, 0 0))");

  // Reversed inner ring (counter-clockwise hole)
  EXPECT_EQ(
      stPolygon(
          "POLYGON((0 0, 0 5, 5 5, 5 0, 0 0), (4 4, 1 4, 1 1, 4 1, 4 4))"),
      "POLYGON ((0 0, 0 5, 5 5, 5 0, 0 0), (4 4, 1 4, 1 1, 4 1, 4 4))");

  // Large coordinate values
  EXPECT_EQ(
      stPolygon(
          "POLYGON((1000000 1000000, 1000000 2000000, 2000000 2000000, 2000000 1000000, 1000000 1000000))"),
      "POLYGON ((1000000 1000000, 1000000 2000000, 2000000 2000000, 2000000 1000000, 1000000 1000000))");

  // Two holes
  EXPECT_EQ(
      stPolygon(
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2), (6 6, 6 8, 8 8, 8 6, 6 6))"),
      "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))");

  // Repeated vertices
  EXPECT_EQ(
      stPolygon(
          "POLYGON((0 0, 0 0, 0 2, 2 2, 2 0, 0 0), (0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5, 0.5 0.5))"),
      "POLYGON ((0 0, 0 0, 0 2, 2 2, 2 0, 0 0), (0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5, 0.5 0.5))");

  // Polygon with negative coordinates
  EXPECT_EQ(
      stPolygon("POLYGON((-5 -5, -5 0, 0 0, 0 -5, -5 -5))"),
      "POLYGON ((-5 -5, -5 0, 0 0, 0 -5, -5 -5))");

  // Empty polygon
  EXPECT_EQ(stPolygon("POLYGON EMPTY"), "POLYGON EMPTY");

  // Not a polygon
  VELOX_ASSERT_USER_THROW(
      stPolygon("LINESTRING(0 0, 1 1)"), "input is not a polygon");
}

TEST_F(GeometryFunctionsTest, stIntersects) {
  const auto stIntersects = [&](const std::optional<std::string>& a,
                                const std::optional<std::string>& b) {
    return evaluateOnce<bool>(
        "ST_Intersects(ST_GeometryFromText(c0), ST_GeometryFromText(c1))",
        a,
        b);
  };

  // Point inside polygon
  EXPECT_EQ(
      stIntersects("POINT(1 1)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"), true);

  // Point outside polygon
  EXPECT_EQ(
      stIntersects("POINT(3 3)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"), false);

  // Point on polygon edge
  EXPECT_EQ(
      stIntersects("POINT(0 1)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"), true);

  // Overlapping polygons
  EXPECT_EQ(
      stIntersects(
          "POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))",
          "POLYGON((2 2, 2 4, 4 4, 4 2, 2 2))"),
      true);

  // Disjoint polygons
  EXPECT_EQ(
      stIntersects(
          "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
          "POLYGON((2 2, 2 3, 3 3, 3 2, 2 2))"),
      false);

  // Line crossing polygon
  EXPECT_EQ(
      stIntersects(
          "LINESTRING(-1 1, 3 1)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"),
      true);

  // Line completely inside polygon
  EXPECT_EQ(
      stIntersects(
          "LINESTRING(1 1, 1.5 1.5)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"),
      true);

  // Line completely outside
  EXPECT_EQ(
      stIntersects(
          "LINESTRING(3 3, 4 4)", "POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))"),
      false);

  // Polygon and its hole
  EXPECT_EQ(
      stIntersects(
          "POLYGON((3 3, 3 5, 5 5, 5 3, 3 3))",
          "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (3 3, 3 5, 5 5, 5 3, 3 3))"),
      true);

  // Touching at a point
  EXPECT_EQ(
      stIntersects(
          "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
          "POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))"),
      true);

  // Touching along an edge
  EXPECT_EQ(
      stIntersects(
          "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
          "POLYGON((1 0, 1 1, 2 1, 2 0, 1 0))"),
      true);

  // Point in multipoint
  EXPECT_EQ(stIntersects("POINT(1 1)", "MULTIPOINT((1 1), (2 2))"), true);

  // MultiLineString overlapping
  EXPECT_EQ(
      stIntersects("MULTILINESTRING((0 0, 2 0))", "LINESTRING(1 0, 1 2)"),
      true);

  // MultiPolygon intersection
  EXPECT_EQ(
      stIntersects(
          "MULTIPOLYGON(((0 0, 0 2, 2 2, 2 0, 0 0)), ((3 3, 3 5, 5 5, 5 3, 3 3)))",
          "POLYGON((1 1, 1 4, 4 4, 4 1, 1 1))"),
      true);

  // GeometryCollection intersecting
  EXPECT_EQ(
      stIntersects(
          "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(1 1, 2 2))", "POINT(1 1)"),
      true);

  // Large coordinates intersection
  EXPECT_EQ(
      stIntersects(
          "POLYGON((1000000 1000000, 1000000 2000000, 2000000 2000000, 2000000 1000000, 1000000 1000000))",
          "POINT(1500000 1500000)"),
      true);
}

#endif
