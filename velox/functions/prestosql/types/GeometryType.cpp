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
#include "velox/functions/prestosql/types/GeometryType.h"

#ifdef VELOX_ENABLE_GEO
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>

#include "velox/functions/prestosql/GeometryUtils.h"
#endif

namespace facebook::velox {

#ifdef VELOX_ENABLE_GEO
class GeometryCastOperator : public exec::CastOperator {
 public:
  // We do not support casting to Geometry from VARBINARY because in the Java
  // implementation, we use a custom format instead of WKB.
  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARBINARY:
        return true;
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  bool isSupportedToType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARBINARY:
        return true;
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (input.typeKind() == TypeKind::VARBINARY) {
      castFromVarbinary(input, context, rows, *result);
    } else if (input.typeKind() == TypeKind::VARCHAR) {
      castFromString(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to Geometry not supported", input.toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (resultType->kind() == TypeKind::VARBINARY) {
      castToVarbinary(input, context, rows, *result);
    } else if (resultType->kind() == TypeKind::VARCHAR) {
      castToString(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from Geometry to {} not supported", resultType->toString());
    }
  }

 private:
  static void castToVarbinary(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometries = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto geometry = geometries->valueAt(row);

      exec::StringWriter<false> varbinaryHexString(flatResult, row);
      auto varbinaryBytes = geometry.data();
      for (size_t i{0}; i < geometry.size(); ++i) {
          unsigned char byte = static_cast<unsigned char>(varbinaryBytes[i]);
          char hexBuf[3];  // 2 digits + null terminator
          std::snprintf(hexBuf, sizeof(hexBuf), "%02X", byte);
          varbinaryHexString.copy_from(hexBuf);
      }
      varbinaryHexString.finalize();
    });
  }

  static void castFromVarbinary(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometryVarbinaryHexStrings = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto varbinaryHexString = geometryVarbinaryHexStrings->valueAt(row);
      auto geosGeometry =
          functions::GeometryUtils::deserialize(varbinaryHexString, true);
      exec::StringWriter<false> geometry(flatResult, row);
      functions::GeometryUtils::serialize(geosGeometry, geometry);
      geometry.finalize();
    });
  }

  static void castToString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometries = input.as<SimpleVector<StringView>>();
    thread_local geos::io::WKTWriter writer;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto geometry = geometries->valueAt(row);
      auto geosGeometry = functions::GeometryUtils::deserialize(geometry);
      exec::StringWriter<false> wktString(flatResult, row);
      wktString.append(writer.write(geosGeometry.get()));
      wktString.finalize();
    });
  }

  static void castFromString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometryStrings = input.as<SimpleVector<StringView>>();
    thread_local geos::io::WKTReader reader;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto wktString = geometryStrings->valueAt(row);
      auto geosGeometry = reader.read(wktString);
      exec::StringWriter<> geometry(flatResult, row);
      functions::GeometryUtils::serialize(geosGeometry, geometry);
      geometry.finalize();
    });
  }
};
#endif

namespace {
class GeometryTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    return GEOMETRY();
  }

  exec::CastOperatorPtr getCastOperator() const override {
#ifdef VELOX_ENABLE_GEO
    return std::make_shared<GeometryCastOperator>();
#else
    return nullptr;
#endif
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& /*config*/) const override {
    return nullptr;
  }
};
} // namespace

void registerGeometryType() {
  registerCustomType(
      "geometry", std::make_unique<const GeometryTypeFactories>());
}

} // namespace facebook::velox
