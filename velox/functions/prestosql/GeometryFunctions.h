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

#pragma once

#include <geos/io/WKBReader.h>
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>

#include "velox/functions/Macros.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/functions/prestosql/GeometryUtils.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

template <typename T>
struct StGeometryFromTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Varchar>& wkt) {
    geos::io::WKTReader reader;
    auto geosGeometry = reader.read(wkt);
    GeometryUtils::serialize(geosGeometry, result);
    return true;
  }
};

template <typename T>
struct StGeomFromBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Varbinary>& wkb) {
    geos::io::WKBReader reader;
    auto geosGeometry = reader.read(
        reinterpret_cast<const unsigned char*>(wkb.data()), wkb.size());
    GeometryUtils::serialize(geosGeometry, result);
    return true;
  }
};

template <typename T>
struct StAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Geometry>& geometryBinary) {
    auto geometry = GeometryUtils::deserialize(geometryBinary);
    geos::io::WKTWriter writer;
    result = writer.write(geometry.get());
    return true;
  }
};

template <typename T>
struct StAsBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varbinary>& result,
      const arg_type<Geometry>& geometryBinary) {
    auto geometry = GeometryUtils::deserialize(geometryBinary);
    geos::io::WKBWriter writer;
    std::ostringstream os;
    writer.write(*geometry, os);
    const auto str = os.str();
    result.resize(str.size());
    std::memcpy(result.data(), str.data(), str.size());
    return true;
  }
};

template <typename T>
struct StContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      bool& result,
      const arg_type<Geometry>& left,
      const arg_type<Geometry>& right) {
    auto leftGeometry = GeometryUtils::deserialize(left);
    auto rightGeometry = GeometryUtils::deserialize(right);
    result = leftGeometry->contains(rightGeometry.get());
  }
};

template <typename T>
struct StWithinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      bool& result,
      const arg_type<Geometry>& left,
      const arg_type<Geometry>& right) {
    auto leftGeometry = GeometryUtils::deserialize(left);
    auto rightGeometry = GeometryUtils::deserialize(right);
    result = leftGeometry->within(rightGeometry.get());
  }
};

template <typename T>
struct StDistanceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      double& result,
      const arg_type<Geometry>& left,
      const arg_type<Geometry>& right) {
    auto leftGeometry = GeometryUtils::deserialize(left);
    auto rightGeometry = GeometryUtils::deserialize(right);
    result = leftGeometry->distance(rightGeometry.get());
  }
};

template <typename T>
struct StIntersectsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      bool& result,
      const arg_type<Geometry>& left,
      const arg_type<Geometry>& right) {
    auto leftGeometry = GeometryUtils::deserialize(left);
    auto rightGeometry = GeometryUtils::deserialize(right);
    result = leftGeometry->intersects(rightGeometry.get());
  }
};

template <typename T>
struct StCentroidFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry) {
    auto geosGeometry = GeometryUtils::deserialize(geometry);
    auto geosGeometryType = geosGeometry->getGeometryType();
    if (geosGeometryType == geos::geom::GEOS_GEOMETRYCOLLECTION) {
      VELOX_USER_FAIL("StCentroidFunction: input is a geometry collection");
      return false;
    }
    if (geosGeometryType == geos::geom::GEOS_POINT) {
      result = geometry;
      return true;
    }

    if (geosGeometry->getNumPoints() == 0) {
      GeometryUtils::serialize(GeometryUtils::createPoint(), result);
      return true;
    }

    GeometryUtils::serialize(geosGeometry->getCentroid(), result);
    return true;
  }
};

template <typename T>
struct StPointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void
  call(out_type<Geometry>& result, double x, double y) {
    auto geometry = GeometryUtils::createPoint(x, y);
    GeometryUtils::serialize(geometry, result);
  }
};

template <typename T>
struct StPolygonFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Varchar>& wkt) {
    thread_local geos::io::WKTReader reader;
    auto geosGeometry = reader.read(wkt);
    if (geosGeometry->getGeometryType() != "Polygon") {
      VELOX_USER_FAIL("StPolygonFunction: input is not a polygon");
      return false;
    }
    GeometryUtils::serialize(geosGeometry, result);
    return true;
  }
};

} // namespace facebook::velox::functions
