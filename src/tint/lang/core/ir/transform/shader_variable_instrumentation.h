// Copyright 2026 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SRC_TINT_LANG_CORE_IR_TRANSFORM_SHADER_VARIABLE_INSTRUMENTATION_H_
#define SRC_TINT_LANG_CORE_IR_TRANSFORM_SHADER_VARIABLE_INSTRUMENTATION_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "src/tint/api/common/binding_point.h"
#include "src/tint/lang/core/ir/validator.h"
#include "src/tint/utils/diagnostic/source.h"
#include "src/tint/utils/result.h"

// Forward declarations.
namespace tint::core::ir {
class Module;
}

namespace tint::core::ir::transform {

/// The capabilities that the ShaderVariableInstrumentation transform can support.
const Capabilities kShaderVariableInstrumentationCapabilities{
    Capability::kAllowDuplicateBindings,
    Capability::kAllow8BitIntegers,
    Capability::kAllow16BitIntegers,
};

/// The scalar type of a value that was captured by the instrumentation.
enum class ShaderVariableInstrumentationScalarType : uint32_t {
    kU32 = 0,
    kI32 = 1,
    kF32 = 2,
};

/// Configuration options for ShaderVariableInstrumentation.
///
/// The transform instruments scalar store instructions (u32, i32, f32) whose
/// destination points at a user variable (typically `function`, `private` or
/// `workgroup` address-space variables) so that the WebGPU debugger can
/// observe how variables change over the course of a shader invocation.
///
/// A new storage buffer is added to the module at @p buffer_binding_point. The
/// buffer has the following logical layout:
///
///     struct TintDebugBuffer {
///       cursor  : atomic<u32>,
///       records : array<u32>,
///     }
///
/// Each instrumented store appends a 2-u32 record:
///
///   records[slot * 2 + 0]:  packed id  (bitfield, see below)
///   records[slot * 2 + 1]:  the stored value, bitcast to u32
///
/// The packed id is a u32 bitfield:
///
///   bits [0:9]   — 10 bits — variable id (0–1023)
///   bits [10:25] — 16 bits — source line number (0–65535)
///   bits [26:29] —  4 bits — @builtin(sample_index) (0–15)
///   bits [30:31] —  2 bits — reserved (zero)
///
/// The variable id and line number are compile-time constants baked in by
/// the transform. The sample index is a per-invocation runtime value read
/// from a `@builtin(sample_index)` parameter that the transform adds to
/// every fragment entry point (reusing an existing one if it already
/// exists). Adding this builtin enables per-sample shading for fragment
/// stages.
///
/// For vertex and compute entry points, the sample index bits are always 0.
///
/// The result of the transform provides a mapping from each assigned
/// variable id back to the variable name, scalar type and source
/// location.
struct ShaderVariableInstrumentationConfig {
    /// The binding point to use for the debug storage buffer added by the
    /// transform.
    BindingPoint buffer_binding_point{};

    /// Binding points whose store instructions should never be instrumented.
    /// Stores whose destination ultimately traces back to a Var whose binding
    /// point is in this set are skipped. This can be used by callers to avoid
    /// instrumenting stores that target resource bindings that the caller
    /// wants to leave untouched.
    std::unordered_set<tint::BindingPoint> skip_bindings{};

    /// If set, only captures stores executed by the fragment invocation whose
    /// `@builtin(position).xy` (truncated to `u32`) matches `{x, y}`.
    ///
    /// When enabled, the transform:
    ///   * adds a `var<private> tint_shader_debug_match : bool = false`
    ///     flag to the module,
    ///   * for every fragment entry point, ensures the entry point has a
    ///     `@builtin(position) : vec4<f32>` parameter (adding one if it
    ///     doesn't already have one) and emits code at the top of the
    ///     entry-point body that assigns
    ///     `tint_shader_debug_match = (u32(pos.x) == x) & (u32(pos.y) == y)`,
    ///   * wraps every instrumented store's append sequence in
    ///     `if (tint_shader_debug_match) { ... }`.
    ///
    /// The flag lives in the `private` address space, so it is per-invocation
    /// and is therefore also `false` for vertex and compute invocations that
    /// happen to share the module. As a result, when this option is set only
    /// the targeted fragment invocation(s) produce any debug records.
    /// Under MSAA, each sample of the same pixel is a separate invocation
    /// that passes the gate; the sample index in the packed id distinguishes
    /// them.
    std::optional<std::array<uint32_t, 2>> target_fragment_coord{};
};

/// Bit layout constants for the packed record id.
/// The packed u32 is: `(sample_index << 26) | (line << 10) | variable_id`.
struct ShaderVariableInstrumentationIdLayout {
    static constexpr uint32_t kVariableIdBits = 10;
    static constexpr uint32_t kLineBits = 16;
    static constexpr uint32_t kSampleIndexBits = 4;

    static constexpr uint32_t kVariableIdShift = 0;
    static constexpr uint32_t kLineShift = kVariableIdBits;
    static constexpr uint32_t kSampleIndexShift = kVariableIdBits + kLineBits;

    static constexpr uint32_t kVariableIdMask = (1u << kVariableIdBits) - 1u;
    static constexpr uint32_t kLineMask = (1u << kLineBits) - 1u;
    static constexpr uint32_t kSampleIndexMask = (1u << kSampleIndexBits) - 1u;
};

/// Information about a single instrumented store.
struct ShaderVariableInstrumentationRecordInfo {
    /// The 10-bit variable id embedded in the packed record id.
    uint32_t variable_id = 0;
    /// The source line number embedded in the packed record id (16-bit,
    /// clamped to 65535).
    uint32_t line = 0;
    /// The scalar type of the captured value (used by the debugger to
    /// re-interpret the u32 record bits).
    ShaderVariableInstrumentationScalarType scalar_type =
        ShaderVariableInstrumentationScalarType::kU32;
    /// The name of the destination variable, if known, otherwise empty.
    std::string variable_name;
    /// The source location of the originating store, if known.
    Source source;
};

/// The result of running ShaderVariableInstrumentation.
struct ShaderVariableInstrumentationResult {
    /// One entry per instrumented store, indexed by the id written to the
    /// debug buffer.
    std::vector<ShaderVariableInstrumentationRecordInfo> records;
};

/// ShaderVariableInstrumentation is a transform used by the WebGPU debugger to
/// observe scalar variable updates inside shaders. For every instrumented
/// store, it emits code that atomically reserves a slot in a storage buffer
/// and writes the assigned store id together with the stored scalar value.
///
/// @param module the module to transform
/// @param config the configuration
/// @returns a @ref ShaderVariableInstrumentationResult on success, or a failure
Result<ShaderVariableInstrumentationResult> ShaderVariableInstrumentation(
    Module& module,
    const ShaderVariableInstrumentationConfig& config);

}  // namespace tint::core::ir::transform

#endif  // SRC_TINT_LANG_CORE_IR_TRANSFORM_SHADER_VARIABLE_INSTRUMENTATION_H_
