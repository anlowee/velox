/*
*  Copyright (c) Facebook, Inc. and its affiliates.
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

#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>

#include "velox/expression/CastExpr.h"
#include "velox/expression/StringWriter.h"
#include "velox/functions/prestosql/geospatial/GeometrySerde.h"
#include "velox/functions/prestosql/geospatial/GeometryUtils.h"
#include "velox/functions/prestosql/types/GeometryCastOperator.h"

#include <io/WKBReader.h>
#include <io/WKBWriter.h>

#include "functions/lib/ToHex.h"
#include "functions/prestosql/URLFunctions.h"

namespace facebook::velox {

bool GeometryCastOperator::isSupportedFromType(const TypePtr& other) const {
  switch (other->kind()) {
    case TypeKind::VARBINARY:
      return true;
    case TypeKind::VARCHAR:
      return true;
    default:
      return false;
  }
}

bool GeometryCastOperator::isSupportedToType(const TypePtr& other) const {
  switch (other->kind()) {
    case TypeKind::VARBINARY:
      return true;
    case TypeKind::VARCHAR:
      return true;
    default:
      return false;
  }
}

void GeometryCastOperator::castTo(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    const TypePtr& resultType,
    VectorPtr& result) const {
  context.ensureWritable(rows, resultType, result);

  if (input.typeKind() == TypeKind::VARBINARY) {
    castFromVarbinary(input, context, rows, *result);
  } else if (input.typeKind() == TypeKind::VARCHAR) {
    castFromVarchar(input, context, rows, *result);
  } else {
    VELOX_UNSUPPORTED(
        "Cast from {} to Geometry not supported", input.toString());
  }
}

void GeometryCastOperator::castFrom(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    const TypePtr& resultType,
    VectorPtr& result) const {
  context.ensureWritable(rows, resultType, result);

  if (resultType->kind() == TypeKind::VARBINARY) {
    castToVarbinary(input, context, rows, *result);
  } else if (resultType->kind() == TypeKind::VARCHAR) {
    castToVarchar(input, context, rows, *result);
  } else {
    VELOX_UNSUPPORTED(
        "Cast from Geometry to {} not supported", resultType->toString());
  }
}

void GeometryCastOperator::castToVarbinary(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.as<FlatVector<StringView>>();
  const auto* geometries = input.as<SimpleVector<StringView>>();

  context.applyToSelectedNoThrow(rows, [&](auto row) {
    const auto geometry = geometries->valueAt(row);

    auto varbinaryHexString = exec::StringWriter(flatResult, row);
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

void GeometryCastOperator::castFromVarbinary(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.as<FlatVector<StringView>>();
  const auto* geometryVarbinaryHexStrings = input.as<SimpleVector<StringView>>();

  context.applyToSelectedNoThrow(rows, [&](auto row) {
    const auto varbinaryHexString = geometryVarbinaryHexStrings->valueAt(row);
    auto geosGeometry =
        functions::geospatial::GeometryDeserializer::deserialize(varbinaryHexString, true);
    auto geometry = exec::StringWriter(flatResult, row);
    functions::geospatial::GeometrySerializer::serialize(*geosGeometry, geometry);
    geometry.finalize();
  });
}

void GeometryCastOperator::castToVarchar(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.as<FlatVector<StringView>>();
  const auto* geometries = input.as<SimpleVector<StringView>>();
  thread_local geos::io::WKTWriter writer;

  context.applyToSelectedNoThrow(rows, [&](auto row) {
    const auto geometry = geometries->valueAt(row);
    auto geosGeometry =
        functions::geospatial::GeometryDeserializer::deserialize(geometry);
    auto wktString = exec::StringWriter(flatResult, row);
    wktString.append(writer.write(geosGeometry.get()));
    wktString.finalize();
  });
}

void GeometryCastOperator::castFromVarchar(
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

    auto geometry = exec::StringWriter(flatResult, row);
    functions::geospatial::GeometrySerializer::serialize(*geosGeometry, geometry);
    geometry.finalize();

    // Here is to get some testing samples to write unit tests
    // auto varbinaryBytes =functions::ToHexUtil::getHexString(geometry.data(), geometry.size());
    // std::cout << "Varbinary hex: " << varbinaryBytes << std::endl;
  });
}

} // namespace facebook::velox
