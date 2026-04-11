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

#include "src/tint/lang/core/ir/transform/shader_variable_instrumentation.h"

#include <utility>

#include "src/tint/lang/core/fluent_types.h"
#include "src/tint/lang/core/ir/access.h"
#include "src/tint/lang/core/ir/builder.h"
#include "src/tint/lang/core/ir/instruction_result.h"
#include "src/tint/lang/core/ir/let.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/store.h"
#include "src/tint/lang/core/ir/validator.h"
#include "src/tint/lang/core/ir/var.h"
#include "src/tint/lang/core/number.h"
#include "src/tint/lang/core/type/atomic.h"
#include "src/tint/lang/core/type/f32.h"
#include "src/tint/lang/core/type/i32.h"
#include "src/tint/lang/core/type/manager.h"
#include "src/tint/lang/core/type/pointer.h"
#include "src/tint/lang/core/type/struct.h"
#include "src/tint/lang/core/type/u32.h"

using namespace tint::core::fluent_types;     // NOLINT
using namespace tint::core::number_suffixes;  // NOLINT

namespace tint::core::ir::transform {

namespace {

/// The index of the `cursor` member within the debug buffer struct.
constexpr uint32_t kCursorMemberIndex = 0;
/// The index of the `records` member within the debug buffer struct.
constexpr uint32_t kRecordsMemberIndex = 1;

/// PIMPL state for the transform.
struct State {
    /// The IR module.
    Module& ir;
    /// The transform configuration.
    const ShaderVariableInstrumentationConfig& config;

    /// The IR builder.
    Builder b{ir};
    /// The type manager.
    core::type::Manager& ty{ir.Types()};

    /// The debug storage buffer variable, created lazily.
    Var* debug_buffer_var = nullptr;
    /// The debug storage buffer struct type, created lazily.
    const core::type::Struct* debug_buffer_struct = nullptr;

    /// The accumulated records, indexed by assigned id.
    std::vector<ShaderVariableInstrumentationRecordInfo> records{};

    /// Walk back through trivial chain instructions (Let, Access) to find the
    /// originating Var for a pointer value.
    /// @returns the Var, or nullptr if one could not be found.
    Var* OriginatingVar(Value* value) {
        Value* current = value;
        while (current != nullptr) {
            auto* result = current->As<InstructionResult>();
            if (result == nullptr) {
                return nullptr;
            }
            auto* inst = result->Instruction();
            if (auto* var = inst->As<Var>()) {
                return var;
            }
            if (auto* access = inst->As<Access>()) {
                current = access->Object();
                continue;
            }
            if (auto* let = inst->As<Let>()) {
                current = let->Value();
                continue;
            }
            return nullptr;
        }
        return nullptr;
    }

    /// @returns true if the given Store instruction should be instrumented.
    bool ShouldInstrument(Store* store) {
        // Only instrument scalar u32/i32/f32 stores. Vectors, matrices, arrays,
        // structs, booleans and f16 are intentionally excluded from v1 so that
        // every record can be represented as a single u32 in the debug buffer.
        const auto* from_type = store->From()->Type();
        if (!from_type->IsAnyOf<core::type::U32, core::type::I32, core::type::F32>()) {
            return false;
        }

        // Skip stores whose destination cannot be traced back to a Var. This
        // keeps the transform conservative and avoids instrumenting exotic
        // pointer plumbing (function parameters, etc.) in v1.
        auto* var = OriginatingVar(store->To());
        if (var == nullptr) {
            return false;
        }

        // Never instrument stores whose destination is a resource binding the
        // caller has asked us to skip.
        if (auto bp = var->BindingPoint()) {
            if (config.skip_bindings.count(*bp) != 0) {
                return false;
            }
            // Stores to storage buffers are the user's final outputs; leave
            // them alone. Tracking them would also risk recursing into the
            // debug buffer we are about to introduce.
            return false;
        }

        // Only instrument stores that target private, function or workgroup
        // address-space variables: these are the "user variables" that the
        // debugger is interested in.
        const auto* var_ptr = var->Result()->Type()->As<core::type::Pointer>();
        if (var_ptr == nullptr) {
            return false;
        }
        switch (var_ptr->AddressSpace()) {
            case core::AddressSpace::kFunction:
            case core::AddressSpace::kPrivate:
            case core::AddressSpace::kWorkgroup:
                return true;
            default:
                return false;
        }
    }

    /// Determine the scalar type tag for a value's type.
    ShaderVariableInstrumentationScalarType ScalarTypeOf(const core::type::Type* type) {
        if (type->Is<core::type::I32>()) {
            return ShaderVariableInstrumentationScalarType::kI32;
        }
        if (type->Is<core::type::F32>()) {
            return ShaderVariableInstrumentationScalarType::kF32;
        }
        return ShaderVariableInstrumentationScalarType::kU32;
    }

    /// Find the destination variable name for a store, tracing through trivial
    /// chain instructions.
    /// @returns the name, or an empty string if none could be determined.
    std::string NameOfDestination(Store* store) {
        auto* var = OriginatingVar(store->To());
        if (var == nullptr) {
            return "";
        }
        auto sym = ir.NameOf(var->Result());
        if (!sym.IsValid()) {
            return "";
        }
        return sym.Name();
    }

    /// Create (once) the debug storage buffer variable and its struct type.
    void EnsureDebugBuffer() {
        if (debug_buffer_var != nullptr) {
            return;
        }
        // struct TintDebugBuffer {
        //   cursor : atomic<u32>,
        //   records : array<u32>,
        // }
        debug_buffer_struct =
            ty.Struct(ir.symbols.New("tint_shader_debug_buffer"),
                      {
                          {ir.symbols.Register("cursor"), ty.atomic<u32>()},
                          {ir.symbols.Register("records"), ty.array<u32>()},
                      });

        b.Append(ir.root_block, [&] {
            debug_buffer_var =
                b.Var("tint_shader_debug_buffer",
                      ty.ptr(core::AddressSpace::kStorage, debug_buffer_struct,
                             core::Access::kReadWrite));
        });
        debug_buffer_var->SetBindingPoint(config.buffer_binding_point.group,
                                          config.buffer_binding_point.binding);
    }

    /// Emit the instrumentation sequence for a single store.
    /// @param store the store being instrumented
    /// @param id the assigned record id
    void EmitInstrumentation(Store* store, uint32_t id) {
        const auto* from_type = store->From()->Type();
        auto* stored_value = store->From();

        b.InsertAfter(store, [&] {
            // slot = atomicAdd(&buffer.cursor, 1u)
            auto* cursor_ptr = b.Access(
                ty.ptr(core::AddressSpace::kStorage, ty.atomic<u32>(), core::Access::kReadWrite),
                debug_buffer_var, u32(kCursorMemberIndex));
            auto* slot =
                b.Call(ty.u32(), core::BuiltinFn::kAtomicAdd, cursor_ptr, 1_u)->Result();

            // idx = slot * 2u
            auto* idx = b.Multiply(slot, 2_u)->Result();
            // idx_plus_one = idx + 1u
            auto* idx_plus_one = b.Add(idx, 1_u)->Result();

            // records[idx] = id
            auto* id_ptr = b.Access(
                ty.ptr(core::AddressSpace::kStorage, ty.u32(), core::Access::kReadWrite),
                debug_buffer_var, u32(kRecordsMemberIndex), idx);
            b.Store(id_ptr, u32(id));

            // records[idx + 1] = bitcast<u32>(stored_value)
            Value* bits = nullptr;
            if (from_type->Is<core::type::U32>()) {
                bits = stored_value;
            } else {
                bits = b.Bitcast(ty.u32(), stored_value)->Result();
            }
            auto* value_ptr = b.Access(
                ty.ptr(core::AddressSpace::kStorage, ty.u32(), core::Access::kReadWrite),
                debug_buffer_var, u32(kRecordsMemberIndex), idx_plus_one);
            b.Store(value_ptr, bits);
        });
    }

    /// Run the transform.
    ShaderVariableInstrumentationResult Run() {
        // First pass: collect all candidate stores before mutating the module.
        // This avoids instrumenting stores we insert ourselves and keeps the
        // instruction-iterator stable while we mutate.
        std::vector<Store*> candidates;
        for (auto* inst : ir.Instructions()) {
            if (auto* store = inst->As<Store>()) {
                if (ShouldInstrument(store)) {
                    candidates.push_back(store);
                }
            }
        }

        if (candidates.empty()) {
            return {};
        }

        // Create the debug buffer. It is important that this happens *after*
        // we have finished the candidate scan, so that the Var and Stores we
        // introduce for the buffer itself are never candidates themselves.
        EnsureDebugBuffer();

        // Second pass: instrument each candidate, assigning a sequential id.
        for (auto* store : candidates) {
            const uint32_t id = static_cast<uint32_t>(records.size());

            ShaderVariableInstrumentationRecordInfo info;
            info.id = id;
            info.scalar_type = ScalarTypeOf(store->From()->Type());
            info.variable_name = NameOfDestination(store);
            info.source = ir.SourceOf(store);
            records.push_back(std::move(info));

            EmitInstrumentation(store, id);
        }

        ShaderVariableInstrumentationResult result;
        result.records = std::move(records);
        return result;
    }
};

}  // namespace

Result<ShaderVariableInstrumentationResult> ShaderVariableInstrumentation(
    Module& ir,
    const ShaderVariableInstrumentationConfig& config) {
    core::ir::AssertValid(ir, kShaderVariableInstrumentationCapabilities,
                          "before core.ShaderVariableInstrumentation");

    State state{ir, config};
    return state.Run();
}

}  // namespace tint::core::ir::transform
