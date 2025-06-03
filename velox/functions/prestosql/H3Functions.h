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

#include <h3/h3api.h>
#include <cmath>

#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"
#include <velox/type/StringView.h>

namespace facebook::velox::functions {

constexpr size_t H3_STRING_LENGTH = 16;

template <typename T>
struct GetHexagonAddrFunction {
    VELOX_DEFINE_FUNCTION_TYPES(T);
  
    FOLLY_ALWAYS_INLINE Status
    call(out_type<Varchar>& result,
         const arg_type<double>& lat,
         const arg_type<double>& lng,
         const arg_type<int64_t>& res) {
        LatLng location;
        location.lat = degsToRads(lat);
        location.lng = degsToRads(lng);
        H3Index indexed;
        if (latLngToCell(&location, res, &indexed) != E_SUCCESS) {
            return Status::UserError(fmt::format(
                "get_hexagon_addr failed, lat={} lng={} res{}", lat, lng, res));
        }
        char h3Str[H3_STRING_LENGTH];
        h3ToString(indexed, h3Str, sizeof(h3Str));
        result.resize(H3_STRING_LENGTH);
        std::memcpy(result.data(), h3Str, H3_STRING_LENGTH);
        return Status::OK();
      }
  };

} // namespace facebook::velox::functions