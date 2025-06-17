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

#include <geos/geom/GeometryFactory.h>
#include <geos/geom/LineString.h>
#include <geos/geom/Point.h>

#include <velox/common/base/Exceptions.h>
#include <velox/type/StringView.h>

namespace facebook::velox::functions {

class GeometryUtils {
 public:
  template <typename StringWriter>
  static void serialize(
      const std::unique_ptr<geos::geom::Geometry>& geometry,
      StringWriter& stringWriter) {
    SliceOutput sliceOutput(stringWriter);
    serialize(geometry.get(), sliceOutput);
  }

  static std::unique_ptr<geos::geom::Geometry> deserialize(
      const StringView& geometryString, bool isFromVarbinary = false) {
    SliceInput sliceInput(geometryString.data(), geometryString.size());
    if (isFromVarbinary) {
      return deserializeWithEsriType(sliceInput);
    }
    return deserialize(sliceInput);
  }

  static std::unique_ptr<geos::geom::Geometry> createPoint() {
    return std::unique_ptr<geos::geom::Point>(
        getGeometryFactory()->createPoint());
  }

  static std::unique_ptr<geos::geom::Geometry> createPoint(double x, double y) {
    geos::geom::Coordinate coord(x, y);
    return std::unique_ptr<geos::geom::Point>(
        getGeometryFactory()->createPoint(coord));
  }

 private:
  enum class GeometrySerializationType : uint8_t {
    POINT = 0,
    MULTI_POINT = 1,
    LINE_STRING = 2,
    MULTI_LINE_STRING = 3,
    POLYGON = 4,
    MULTI_POLYGON = 5,
    GEOMETRY_COLLECTION = 6,
    ENVELOPE = 7
  };

  enum class EsriShapeType : int32_t {
    POINT = 1,
    POLYLINE = 3,
    POLYGON = 5,
    MULTI_POINT = 8
  };

  class SliceInput {
   public:
    SliceInput(const void* rawData, size_t dataSize)
        : data_(reinterpret_cast<const uint8_t*>(rawData)),
          size_(dataSize),
          position_(0) {}

    uint8_t readByte() {
      return read<uint8_t>();
    }

    int16_t readShort() {
      return read<int16_t>();
    }

    int32_t readInt() {
      return read<int32_t>();
    }

    int32_t readLong() {
      return read<int64_t>();
    }

    float readFloat() {
      return read<float>();
    }

    double readDouble() {
      return read<double>();
    }

    size_t remaining() const {
      return size_ - position_;
    }

    void skip(size_t size) {
      if (position_ + size > size_) {
        VELOX_USER_FAIL("SliceInput read exceeds buffer size.");
      }
      position_ += size;
    }

   private:
    template <typename T>
    T read() {
      if (position_ + sizeof(T) > size_) {
        VELOX_USER_FAIL("SliceInput read exceeds buffer size.");
      }
      T value;
      std::memcpy(&value, &data_[position_], sizeof(T));
      position_ += sizeof(T);
      return value;
    }

    const uint8_t* data_;
    size_t size_;
    size_t position_;
  };

  template <typename StringWriter>
  class SliceOutput {
   public:
    SliceOutput(StringWriter& stringWriter) : stringWriter_(stringWriter) {}

    SliceOutput() = delete;

    void writeByte(uint8_t value) {
      write<uint8_t>(value);
    }

    void writeShort(int16_t value) {
      write<int16_t>(value);
    }

    void writeInt(int32_t value) {
      write<int32_t>(value);
    }

    void writeLong(int64_t value) {
      write<int64_t>(value);
    }

    void writeFloat(float value) {
      write<float>(value);
    }

    void writeDouble(double value) {
      write<double>(value);
    }

    void writeBytes(const char* data, size_t size) {
      stringWriter_.append(std::string_view(data, size));
    }

   private:
    template <typename T>
    void write(const T& value) {
      stringWriter_.append(
          std::string_view(reinterpret_cast<const char*>(&value), sizeof(T)));
    }

    StringWriter& stringWriter_;
  };

  template <typename T>
  static void serialize(
      const geos::geom::Geometry* geometry,
      SliceOutput<T>& sliceOutput) {
    auto geometryType = geometry->getGeometryTypeId();
    switch (geometryType) {
      case geos::geom::GEOS_POINT:
        writePoint(geometry, sliceOutput);
        break;
      case geos::geom::GEOS_MULTIPOINT:
        writeMultiPoint(geometry, sliceOutput);
        break;
      case geos::geom::GEOS_LINESTRING:
      case geos::geom::GEOS_LINEARRING:
        writePolyline(geometry, sliceOutput, false);
        break;
      case geos::geom::GEOS_MULTILINESTRING:
        writePolyline(geometry, sliceOutput, true);
        break;
      case geos::geom::GEOS_POLYGON:
        writePolygon(geometry, sliceOutput, false);
        break;
      case geos::geom::GEOS_MULTIPOLYGON:
        writePolygon(geometry, sliceOutput, true);
        break;
      case geos::geom::GEOS_GEOMETRYCOLLECTION:
        writeGeometryCollection(geometry, sliceOutput);
        break;
      default:
        VELOX_USER_FAIL(
            "Unrecognized geometry type: {}", static_cast<int>(geometryType));
        break;
    }
  }

  static std::unique_ptr<geos::geom::Geometry> deserialize(
      SliceInput& sliceInput) {
    auto geometryType =
        static_cast<GeometrySerializationType>(sliceInput.readByte());
    switch (geometryType) {
      case GeometrySerializationType::POINT:
        return readPoint(sliceInput, 0);
      case GeometrySerializationType::MULTI_POINT:
        skipEsriType(sliceInput);
        skipEnvelope(sliceInput);
        return readMultiPoint(sliceInput, 0);
      case GeometrySerializationType::LINE_STRING:
        skipEsriType(sliceInput);
        skipEnvelope(sliceInput);
        return readPolyline(sliceInput, false, 0);
      case GeometrySerializationType::MULTI_LINE_STRING:
        skipEsriType(sliceInput);
        skipEnvelope(sliceInput);
        return readPolyline(sliceInput, true, 0);
      case GeometrySerializationType::POLYGON:
        skipEsriType(sliceInput);
        skipEnvelope(sliceInput);
        return readPolygon(sliceInput, false, 0);
      case GeometrySerializationType::MULTI_POLYGON:
        skipEsriType(sliceInput);
        skipEnvelope(sliceInput);
        return readPolygon(sliceInput, true, 0);
      case GeometrySerializationType::ENVELOPE:
        return readEnvelope(sliceInput);
      case GeometrySerializationType::GEOMETRY_COLLECTION:
        return readGeometryCollection(sliceInput);
      default:
        VELOX_USER_FAIL(
            "Unrecognized geometry type: {}",
            static_cast<uint8_t>(geometryType));
        break;
    }
    return nullptr;
  }

  static std::unique_ptr<geos::geom::Geometry> deserializeWithEsriType(
      SliceInput& sliceInput) {
    int SRID{0};
    SRID = sliceInput.readInt();
    // Skip 1-byte Geometry type but use 4-bytes Esri type
    sliceInput.readByte();
    auto esriType =
        static_cast<EsriShapeType>(sliceInput.readInt());
    switch (esriType) {
      case EsriShapeType::POINT:
        return readPoint(sliceInput, SRID);
      case EsriShapeType::MULTI_POINT:
        skipEnvelope(sliceInput);
        return readMultiPoint(sliceInput, SRID);
      case EsriShapeType::POLYLINE:
        skipEnvelope(sliceInput);
        return readPolyline(sliceInput, false, SRID);
      case EsriShapeType::POLYGON:
        skipEnvelope(sliceInput);
        return readPolygon(sliceInput, false, SRID);
      default:
        VELOX_USER_FAIL(
            "Unrecognized esri type: {}",
            static_cast<int32_t>(esriType));
        break;
    }
    return nullptr;
  }

  static bool isEsriNaN(double d) {
    auto translateFromAVNaN = [](double value) -> double { return value; };
    return std::isnan(d) || std::isnan(translateFromAVNaN(d));
  }

  static bool isClockwise(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      int start,
      int end) {
    double sum = 0.0;
    for (int i = start; i < end - 1; i++) {
      const auto& p1 = coordinates->getAt(i);
      const auto& p2 = coordinates->getAt(i + 1);
      sum += (p2.x - p1.x) * (p2.y + p1.y);
    }
    return sum > 0.0;
  }

  static void reverse(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      int start,
      int end) {
    for (int i = 0; i < (end - start) / 2; ++i) {
      auto temp = coordinates->getAt(start + i);
      coordinates->setAt(coordinates->getAt(end - 1 - i), start + i);
      coordinates->setAt(temp, end - 1 - i);
    }
  }

  template <typename T>
  static void writeEnvelope(
      const geos::geom::Geometry* geometry,
      SliceOutput<T>& output) {
    if (geometry->isEmpty()) {
      output.writeDouble(std::nan(""));
      output.writeDouble(std::nan(""));
      output.writeDouble(std::nan(""));
      output.writeDouble(std::nan(""));
      return;
    }

    auto envelope = geometry->getEnvelopeInternal();
    output.writeDouble(envelope->getMinX());
    output.writeDouble(envelope->getMinY());
    output.writeDouble(envelope->getMaxX());
    output.writeDouble(envelope->getMaxY());
  }

  template <typename T>
  static void writeCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coords,
      SliceOutput<T>& output) {
    for (size_t i = 0; i < coords->size(); ++i) {
      output.writeDouble(coords->getX(i));
      output.writeDouble(coords->getY(i));
    }
  }

  template <typename T>
  static void writePoint(
      const geos::geom::Geometry* point,
      SliceOutput<T>& output) {
    output.writeByte(static_cast<uint8_t>(GeometrySerializationType::POINT));
    if (!point->isEmpty()) {
      writeCoordinates(point->getCoordinates(), output);
    } else {
      output.writeDouble(std::nan(""));
      output.writeDouble(std::nan(""));
    }
  }

  template <typename T>
  static void writeMultiPoint(
      const geos::geom::Geometry* geometry,
      SliceOutput<T>& output) {
    output.writeByte(
        static_cast<uint8_t>(GeometrySerializationType::MULTI_POINT));
    output.writeInt(static_cast<int32_t>(EsriShapeType::MULTI_POINT));
    writeEnvelope(geometry, output);
    output.writeInt(geometry->getNumPoints());
    writeCoordinates(geometry->getCoordinates(), output);
  }

  template <typename T>
  static void writePolyline(
      const geos::geom::Geometry* geometry,
      SliceOutput<T>& output,
      bool multiType) {
    int numParts;
    int numPoints = geometry->getNumPoints();

    if (multiType) {
      numParts = geometry->getNumGeometries();
      output.writeByte(
          static_cast<uint8_t>(GeometrySerializationType::MULTI_LINE_STRING));
    } else {
      numParts = (numPoints > 0) ? 1 : 0;
      output.writeByte(
          static_cast<uint8_t>(GeometrySerializationType::LINE_STRING));
    }

    output.writeInt(static_cast<int32_t>(EsriShapeType::POLYLINE));

    writeEnvelope(geometry, output);

    output.writeInt(numParts);
    output.writeInt(numPoints);

    int partIndex = 0;
    for (int i = 0; i < numParts; ++i) {
      output.writeInt(partIndex);
      partIndex += geometry->getGeometryN(i)->getNumPoints();
    }

    if (multiType) {
      for (int i = 0; i < numParts; ++i) {
        const auto* part = geometry->getGeometryN(i);
        writeCoordinates(part->getCoordinates(), output);
      }
    } else {
      writeCoordinates(geometry->getCoordinates(), output);
    }
  }

  template <typename T>
  static void writePolygon(
      const geos::geom::Geometry* geometry,
      SliceOutput<T>& output,
      bool multitype) {
    int numGeometries = geometry->getNumGeometries();
    int numParts = 0;
    int numPoints = geometry->getNumPoints();

    for (int i = 0; i < numGeometries; i++) {
      auto polygon =
          dynamic_cast<const geos::geom::Polygon*>(geometry->getGeometryN(i));
      if (polygon && polygon->getNumPoints() > 0) {
        numParts += polygon->getNumInteriorRing() + 1;
      }
    }

    if (multitype) {
      output.writeByte(
          static_cast<uint8_t>(GeometrySerializationType::MULTI_POLYGON));
    } else {
      output.writeByte(
          static_cast<uint8_t>(GeometrySerializationType::POLYGON));
    }

    output.writeInt(static_cast<int32_t>(EsriShapeType::POLYGON));
    writeEnvelope(geometry, output);

    output.writeInt(numParts);
    output.writeInt(numPoints);

    if (numParts == 0) {
      return;
    }

    std::vector<int> partIndexes(numParts);
    std::vector<bool> shellPart(numParts);

    int currentPart = 0;
    int currentPoint = 0;
    for (int i = 0; i < numGeometries; i++) {
      const geos::geom::Polygon* polygon =
          dynamic_cast<const geos::geom::Polygon*>(geometry->getGeometryN(i));

      partIndexes[currentPart] = currentPoint;
      shellPart[currentPart] = true;
      currentPart++;
      currentPoint += polygon->getExteriorRing()->getNumPoints();

      int holesCount = polygon->getNumInteriorRing();
      for (int holeIndex = 0; holeIndex < holesCount; holeIndex++) {
        partIndexes[currentPart] = currentPoint;
        shellPart[currentPart] = false;
        currentPart++;
        currentPoint += polygon->getInteriorRingN(holeIndex)->getNumPoints();
      }
    }

    for (int partIndex : partIndexes) {
      output.writeInt(partIndex);
    }

    auto coordinates = geometry->getCoordinates();
    canonicalizePolygonCoordinates(coordinates, partIndexes, shellPart);
    writeCoordinates(coordinates, output);
  }

  static void canonicalizePolygonCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      const std::vector<int>& partIndexes,
      const std::vector<bool>& shellPart) {
    for (size_t part = 0; part < partIndexes.size() - 1; part++) {
      canonicalizePolygonCoordinates(
          coordinates,
          partIndexes[part],
          partIndexes[part + 1],
          shellPart[part]);
    }
    if (!partIndexes.empty()) {
      canonicalizePolygonCoordinates(
          coordinates,
          partIndexes.back(),
          coordinates->size(),
          shellPart.back());
    }
  }

  static void canonicalizePolygonCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      int start,
      int end,
      bool isShell) {
    bool isClockwiseFlag = isClockwise(coordinates, start, end);

    if ((isShell && !isClockwiseFlag) || (!isShell && isClockwiseFlag)) {
      reverse(coordinates, start, end);
    }
  }

  template <typename T>
  static void writeGeometryCollection(
      const geos::geom::Geometry* collection,
      SliceOutput<T>& output) {
    output.writeByte(
        static_cast<uint8_t>(GeometrySerializationType::GEOMETRY_COLLECTION));

    for (size_t geometryIndex = 0;
         geometryIndex < collection->getNumGeometries();
         ++geometryIndex) {
      auto* geometry = collection->getGeometryN(geometryIndex);
      // Use a temporary buffer to serialize the geometry and calculate its
      // length
      std::string tempBuffer;
      SliceOutput tempOutput(tempBuffer);

      serialize(geometry, tempOutput);

      int32_t length = static_cast<int32_t>(tempBuffer.size());
      output.writeInt(length);
      output.writeBytes(tempBuffer.data(), tempBuffer.size());
    }
  }

  static geos::geom::GeometryFactory* getGeometryFactory(int SRID = 0) {
    auto precisionModel = std::make_unique<geos::geom::PrecisionModel>(geos::geom::PrecisionModel::FLOATING);
    thread_local static geos::geom::GeometryFactory::Ptr geometryFactory
      = geos::geom::GeometryFactory::create(precisionModel.get(), SRID);
    return geometryFactory.get();
  }

  static void skipEsriType(SliceInput& input) {
    input.skip(sizeof(int32_t)); // Esri type is an integer
  }

  static void skipEnvelope(SliceInput& input) {
    input.skip(
        4 * sizeof(double)); // Envelopes are 4 doubles (minX, minY, maxX, maxY)
  }

  static geos::geom::Coordinate readCoordinate(SliceInput& input) {
    double x = input.readDouble();
    double y = input.readDouble();
    return {x, y};
  }

  static std::unique_ptr<geos::geom::CoordinateSequence> readCoordinates(
      SliceInput& input,
      int count) {
    auto coords = std::make_unique<geos::geom::CoordinateSequence>(count, 2);
    for (int i = 0; i < count; ++i) {
      coords->setAt(readCoordinate(input), i);
    }
    return coords;
  }

  static std::unique_ptr<geos::geom::Point> readPoint(SliceInput& input, int SRID) {
    geos::geom::Coordinate coordinate = readCoordinate(input);
    if (std::isnan(coordinate.x) || std::isnan(coordinate.y)) {
      return getGeometryFactory(SRID)->createPoint();
    }
    return std::unique_ptr<geos::geom::Point>(
        getGeometryFactory(SRID)->createPoint(coordinate));
  }

  static std::unique_ptr<geos::geom::Geometry> readMultiPoint(
      SliceInput& input,
      int SRID) {
    int pointCount = input.readInt();
    auto coords = readCoordinates(input, pointCount);
    std::vector<std::unique_ptr<geos::geom::Point>> points;
    for (size_t i = 0; i < coords->size(); ++i) {
      points.push_back(
          std::unique_ptr<geos::geom::Point>(getGeometryFactory(SRID)->createPoint(
              geos::geom::Coordinate(coords->getX(i), coords->getY(i)))));
    }
    return getGeometryFactory(SRID)->createMultiPoint(std::move(points));
  }

  static std::unique_ptr<geos::geom::Geometry> readPolyline(
      SliceInput& input,
      bool multiType,
      int SRID) {
    int partCount = input.readInt();

    if (partCount == 0) {
      if (multiType) {
        return getGeometryFactory(SRID)->createMultiLineString();
      }
      return getGeometryFactory(SRID)->createLineString();
    }

    int pointCount = input.readInt();
    std::vector<int> startIndexes(partCount);

    for (int i = 0; i < partCount; ++i) {
      startIndexes[i] = input.readInt();
    }

    std::vector<int> partLengths(partCount);
    if (partCount > 1) {
      partLengths[0] = startIndexes[1];
      for (int i = 1; i < partCount - 1; ++i) {
        partLengths[i] = startIndexes[i + 1] - startIndexes[i];
      }
    }
    partLengths[partCount - 1] = pointCount - startIndexes[partCount - 1];

    std::vector<std::unique_ptr<geos::geom::LineString>> lineStrings;
    for (int i = 0; i < partCount; ++i) {
      lineStrings.push_back(getGeometryFactory(SRID)->createLineString(
          readCoordinates(input, partLengths[i])));
    }

    if (multiType) {
      return getGeometryFactory(SRID)->createMultiLineString(std::move(lineStrings));
    }

    if (lineStrings.size() != 1) {
      VELOX_USER_FAIL(
          "Expected a single LineString for non-multitype polyline.");
    }

    return std::move(lineStrings[0]);
  }

  static std::unique_ptr<geos::geom::Geometry> readPolygon(
      SliceInput& input,
      bool multiType,
      int SRID) {
    int partCount = input.readInt();
    if (partCount == 0) {
      if (multiType) {
        return getGeometryFactory(SRID)->createMultiPolygon();
      }
      return getGeometryFactory(SRID)->createPolygon();
    }

    int pointCount = input.readInt();
    std::vector<int> startIndexes(partCount);
    for (int i = 0; i < partCount; i++) {
      startIndexes[i] = input.readInt();
    }

    std::vector<int> partLengths(partCount);
    if (partCount > 1) {
      partLengths[0] = startIndexes[1];
      for (int i = 1; i < partCount - 1; i++) {
        partLengths[i] = startIndexes[i + 1] - startIndexes[i];
      }
    }
    partLengths[partCount - 1] = pointCount - startIndexes[partCount - 1];

    std::unique_ptr<geos::geom::LinearRing> shell = nullptr;
    std::vector<std::unique_ptr<geos::geom::LinearRing>> holes;
    std::vector<std::unique_ptr<geos::geom::Polygon>> polygons;

    for (int i = 0; i < partCount; i++) {
      auto coordinates = readCoordinates(input, partLengths[i]);

      if (isClockwise(coordinates, 0, coordinates->size())) {
        if (shell) {
          polygons.push_back(getGeometryFactory(SRID)->createPolygon(
              std::move(shell), std::move(holes)));
          holes.clear();
        }
        shell = getGeometryFactory(SRID)->createLinearRing(std::move(coordinates));
      } else {
        holes.push_back(
            getGeometryFactory(SRID)->createLinearRing(std::move(coordinates)));
      }
    }
    polygons.push_back(
        getGeometryFactory(SRID)->createPolygon(std::move(shell), std::move(holes)));

    if (multiType) {
      return getGeometryFactory(SRID)->createMultiPolygon(std::move(polygons));
    }

    if (polygons.size() != 1) {
      VELOX_USER_FAIL("Expected exactly one polygon, but found multiple.");
    }
    return std::move(polygons[0]);
  }

  static std::unique_ptr<geos::geom::Geometry> readEnvelope(SliceInput& input) {
    double xMin = input.readDouble();
    double yMin = input.readDouble();
    double xMax = input.readDouble();
    double yMax = input.readDouble();

    if (isEsriNaN(xMin) || isEsriNaN(yMin) || isEsriNaN(xMax) ||
        isEsriNaN(yMax)) {
      return getGeometryFactory(0)->createPolygon();
    }

    auto coordinates = std::make_unique<geos::geom::CoordinateSequence>();
    coordinates->add(geos::geom::Coordinate(xMin, yMin));
    coordinates->add(geos::geom::Coordinate(xMin, yMax));
    coordinates->add(geos::geom::Coordinate(xMax, yMax));
    coordinates->add(geos::geom::Coordinate(xMax, yMin));
    coordinates->add(geos::geom::Coordinate(xMin, yMin)); // Close the ring

    auto shell = getGeometryFactory(0)->createLinearRing(std::move(coordinates));
    return getGeometryFactory(0)->createPolygon(std::move(shell), {});
  }

  static std::unique_ptr<geos::geom::Geometry> readGeometryCollection(
      SliceInput& input) {
    std::vector<std::unique_ptr<geos::geom::Geometry>> geometries;

    while (input.remaining() > 0) {
      // Skip the length field
      input.readInt();
      geometries.push_back(deserialize(input));
    }
    std::vector<const geos::geom::Geometry*> rawGeometries;
    for (const auto& geometry : geometries) {
      rawGeometries.push_back(geometry.get());
    }

    return std::unique_ptr<geos::geom::GeometryCollection>(
        getGeometryFactory(0)->createGeometryCollection(rawGeometries));
  }
};
} // namespace facebook::velox::functions
