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
#include "src/tint/lang/core/ir/function.h"
#include "src/tint/lang/core/ir/function_param.h"
#include "src/tint/lang/core/ir/if.h"
#include "src/tint/lang/core/ir/instruction_result.h"
#include "src/tint/lang/core/ir/let.h"
#include "src/tint/lang/core/ir/loop.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/return.h"
#include "src/tint/lang/core/ir/store.h"
#include "src/tint/lang/core/ir/switch.h"
#include "src/tint/lang/core/ir/user_call.h"
#include "src/tint/lang/core/ir/validator.h"
#include "src/tint/lang/core/ir/var.h"
#include "src/tint/lang/core/number.h"
#include "src/tint/lang/core/type/atomic.h"
#include "src/tint/lang/core/type/f32.h"
#include "src/tint/lang/core/type/i32.h"
#include "src/tint/lang/core/type/manager.h"
#include "src/tint/lang/core/type/matrix.h"
#include "src/tint/lang/core/type/pointer.h"
#include "src/tint/lang/core/type/struct.h"
#include "src/tint/lang/core/type/bool.h"
#include "src/tint/lang/core/type/i8.h"
#include "src/tint/lang/core/type/u8.h"
#include "src/tint/lang/core/type/u16.h"
#include "src/tint/lang/core/type/u32.h"
#include "src/tint/lang/core/type/u64.h"
#include "src/tint/lang/core/type/vector.h"

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

    /// When `target_fragment_coord` is set, the private var that holds whether
    /// the current invocation should record debug data. Created lazily.
    Var* match_var = nullptr;

    /// Private var holding the per-invocation `@builtin(sample_index)` value
    /// so it is accessible from non-entry-point functions. Created lazily.
    Var* sample_index_var = nullptr;

    /// Map from Var* to the assigned variable_id. Each unique Var gets one
    /// id, shared across all stores to that variable.
    Hashmap<Var*, uint32_t, 64> var_id_map{};

    /// The accumulated variable info entries, indexed by variable_id.
    std::vector<ShaderVariableInstrumentationVariableInfo> variables{};

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

    /// @returns true if a type is a supported scalar for instrumentation.
    static bool IsSupportedScalar(const core::type::Type* t) {
        return t->IsAnyOf<core::type::Bool, core::type::I32, core::type::U32,
                          core::type::F32, core::type::F16, core::type::I8,
                          core::type::U8, core::type::U16, core::type::U64>();
    }

    /// @returns true if the given Store instruction should be instrumented.
    bool ShouldInstrument(Store* store) {
        const auto* from_type = store->From()->Type();

        // Scalars.
        if (IsSupportedScalar(from_type)) {
            // fall through to variable checks below
        } else if (auto* vec = from_type->As<core::type::Vector>()) {
            // Vectors of supported element types.
            if (!IsSupportedScalar(vec->Type())) {
                return false;
            }
            // Only vec2/vec3/vec4 of i32/u32/f32/f16 are encoded; other
            // element types (bool, i8, u8, u16, u64) have no vector tags.
            if (!vec->Type()->IsAnyOf<core::type::I32, core::type::U32,
                                      core::type::F32, core::type::F16>()) {
                return false;
            }
        } else if (auto* mat = from_type->As<core::type::Matrix>()) {
            // Only f32 matrices have tags.
            if (!mat->Type()->Is<core::type::F32>()) {
                return false;
            }
        } else {
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

    /// Map a scalar IR type to a data-type tag.
    static ShaderVariableInstrumentationDataType ScalarDataType(const core::type::Type* t) {
        using DT = ShaderVariableInstrumentationDataType;
        if (t->Is<core::type::Bool>()) return DT::kBool;
        if (t->Is<core::type::I32>()) return DT::kI32;
        if (t->Is<core::type::U32>()) return DT::kU32;
        if (t->Is<core::type::F32>()) return DT::kF32;
        if (t->Is<core::type::F16>()) return DT::kF16;
        if (t->Is<core::type::I8>()) return DT::kI8;
        if (t->Is<core::type::U8>()) return DT::kU8;
        if (t->Is<core::type::U16>()) return DT::kU16;
        if (t->Is<core::type::U64>()) return DT::kU64;
        return DT::kU32;  // fallback
    }

    /// Map any supported IR type to the data-type tag stored in the packed
    /// header. Handles scalars, vectors, and matrices.
    ShaderVariableInstrumentationDataType DataTypeOf(const core::type::Type* type) {
        using DT = ShaderVariableInstrumentationDataType;

        if (auto* vec = type->As<core::type::Vector>()) {
            auto w = vec->Width();
            auto* el = vec->Type();
            // vec{2,3,4} × {i32, u32, f32, f16}
            if (el->Is<core::type::I32>()) return static_cast<DT>(9 + (w - 2));   // 9,10,11
            if (el->Is<core::type::U32>()) return static_cast<DT>(12 + (w - 2));  // 12,13,14
            if (el->Is<core::type::F32>()) return static_cast<DT>(15 + (w - 2));  // 15,16,17
            if (el->Is<core::type::F16>()) return static_cast<DT>(18 + (w - 2));  // 18,19,20
        }

        if (auto* mat = type->As<core::type::Matrix>()) {
            // mat CxR f32, encoded as 21 + (C-2)*3 + (R-2)
            auto c = mat->Columns();
            auto r = mat->Rows();
            return static_cast<DT>(21 + (c - 2) * 3 + (r - 2));
        }

        return ScalarDataType(type);
    }

    /// Create (once) the `var<private> tint_shader_debug_match : bool = false`
    /// flag used to filter by fragment coordinate.
    void EnsureMatchVar() {
        if (match_var != nullptr) {
            return;
        }
        b.Append(ir.root_block, [&] {
            match_var = b.Var("tint_shader_debug_match",
                              ty.ptr(core::AddressSpace::kPrivate, ty.bool_(),
                                     core::Access::kReadWrite));
            match_var->SetInitializer(b.Constant(false));
        });
    }

    /// Create (once) the `var<private> tint_sample_index : u32 = 0u`
    /// variable used to forward the sample_index builtin to non-entry-point
    /// functions.
    void EnsureSampleIndexVar() {
        if (sample_index_var != nullptr) {
            return;
        }
        b.Append(ir.root_block, [&] {
            sample_index_var = b.Var("tint_sample_index",
                                     ty.ptr(core::AddressSpace::kPrivate, ty.u32(),
                                            core::Access::kReadWrite));
            sample_index_var->SetInitializer(b.Constant(0_u));
        });
    }

    /// Find the `@builtin(position)` parameter of a function, adding one if
    /// none exists.
    FunctionParam* GetOrAddPositionParam(Function* func) {
        for (auto* param : func->Params()) {
            if (param->Builtin() == core::BuiltinValue::kPosition) {
                return param;
            }
        }
        auto* pos = b.FunctionParam("tint_frag_coord", ty.vec4<f32>());
        pos->SetBuiltin(core::BuiltinValue::kPosition);
        func->AppendParam(pos);
        return pos;
    }

    /// Find the `@builtin(sample_index)` parameter of a function, adding one
    /// if none exists.
    FunctionParam* GetOrAddSampleIndexParam(Function* func) {
        for (auto* param : func->Params()) {
            if (param->Builtin() == core::BuiltinValue::kSampleIndex) {
                return param;
            }
        }
        auto* si = b.FunctionParam("tint_sample_idx", ty.u32());
        si->SetBuiltin(core::BuiltinValue::kSampleIndex);
        func->AppendParam(si);
        return si;
    }

    /// Insert-at-top helper. Inserts instructions produced by @p cb before
    /// any existing instructions in @p func's body.
    template <typename CB>
    void InsertAtTopOfFunction(Function* func, CB&& cb) {
        auto* body = func->Block();
        if (body->IsEmpty()) {
            b.Append(body, cb);
        } else {
            b.InsertBefore(body->Front(), cb);
        }
    }

    /// Set up a fragment entry point:
    ///   1. Store `@builtin(sample_index)` into the private var.
    ///   2. If `target_fragment_coord` is set, compute the match flag.
    void SetupFragmentEntryPoint(Function* func) {
        auto* si_param = GetOrAddSampleIndexParam(func);

        InsertAtTopOfFunction(func, [&] {
            b.Store(sample_index_var, si_param);

            if (config.target_fragment_coord.has_value()) {
                const auto& xy = *config.target_fragment_coord;
                auto* pos = GetOrAddPositionParam(func);
                auto* px = b.Access(ty.f32(), pos, 0_u)->Result();
                auto* py = b.Access(ty.f32(), pos, 1_u)->Result();
                auto* uxv = b.Convert(ty.u32(), px)->Result();
                auto* uyv = b.Convert(ty.u32(), py)->Result();
                auto* eq_x = b.Equal(uxv, u32(xy[0]))->Result();
                auto* eq_y = b.Equal(uyv, u32(xy[1]))->Result();
                auto* matches = b.And(eq_x, eq_y)->Result();
                b.Store(match_var, matches);
            }
        });
    }

    /// Walk the module and set up every fragment entry point.
    void SetupAllFragmentEntryPoints() {
        for (auto* func : ir.functions) {
            if (func->IsFragment()) {
                SetupFragmentEntryPoint(func);
            }
        }
    }

    /// Create (once) the debug storage buffer variable and its struct type.
    void EnsureDebugBuffer() {
        if (debug_buffer_var != nullptr) {
            return;
        }
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

    /// Convert a single scalar element to a u32 word.
    /// @param elem the scalar value
    /// @param elem_type the element's IR type
    /// @returns a u32 Value
    Value* ScalarElementToU32(Value* elem, const core::type::Type* elem_type) {
        if (elem_type->Is<core::type::U32>()) {
            return elem;
        }
        if (elem_type->IsAnyOf<core::type::I32, core::type::F32>()) {
            return b.Bitcast(ty.u32(), elem)->Result();
        }
        if (elem_type->Is<core::type::Bool>()) {
            return b.Call(ty.u32(), core::BuiltinFn::kSelect, 0_u, 1_u, elem)->Result();
        }
        if (elem_type->Is<core::type::F16>()) {
            return b.Bitcast(ty.u32(), b.Convert(ty.f32(), elem))->Result();
        }
        if (elem_type->Is<core::type::I8>()) {
            return b.Bitcast(ty.u32(), b.Convert(ty.i32(), elem))->Result();
        }
        if (elem_type->IsAnyOf<core::type::U8, core::type::U16>()) {
            return b.Convert(ty.u32(), elem)->Result();
        }
        // u64 → 2 words; handled separately.
        return elem;
    }

    /// Convert a loaded value of any instrumented type to u32 data words.
    /// @param loaded the loaded value
    /// @param from_type the IR type of the loaded value
    /// @param[out] words filled with data word Values
    void ValueToDataWords(Value* loaded, const core::type::Type* from_type,
                          Vector<Value*, 16>& words) {
        // Scalar
        if (IsSupportedScalar(from_type)) {
            if (from_type->Is<core::type::U64>()) {
                auto* pair = b.Bitcast(ty.vec2<u32>(), loaded)->Result();
                words.Push(b.Access(ty.u32(), pair, 0_u)->Result());
                words.Push(b.Access(ty.u32(), pair, 1_u)->Result());
            } else {
                words.Push(ScalarElementToU32(loaded, from_type));
            }
            return;
        }

        // Vector
        if (auto* vec = from_type->As<core::type::Vector>()) {
            auto* el_type = vec->Type();
            for (uint32_t i = 0; i < vec->Width(); i++) {
                auto* comp = b.Access(el_type, loaded, u32(i))->Result();
                words.Push(ScalarElementToU32(comp, el_type));
            }
            return;
        }

        // Matrix — column-major: iterate columns, then rows.
        if (auto* mat = from_type->As<core::type::Matrix>()) {
            auto* el_type = mat->Type();  // scalar element type (f32)
            auto* col_type = mat->ColumnType();
            for (uint32_t c = 0; c < mat->Columns(); c++) {
                auto* col = b.Access(col_type, loaded, u32(c))->Result();
                for (uint32_t r = 0; r < mat->Rows(); r++) {
                    auto* elem = b.Access(el_type, col, u32(r))->Result();
                    words.Push(ScalarElementToU32(elem, el_type));
                }
            }
            return;
        }
    }

    /// Emit the body of the instrumentation append-record sequence into the
    /// builder's current insertion point.
    /// @param store the store being instrumented
    /// @param variable_id the 10-bit variable id
    /// @param line the 13-bit source line number
    /// @param dt the data type tag
    void EmitAppendRecord(Store* store,
                          uint32_t variable_id,
                          uint32_t line,
                          ShaderVariableInstrumentationDataType dt) {
        using L = ShaderVariableInstrumentationIdLayout;

        // Read the value back from the destination pointer.
        auto* loaded = b.Load(store->To())->Result();

        // Convert the loaded value to u32 data words.
        Vector<Value*, 16> data_words;
        ValueToDataWords(loaded, store->From()->Type(), data_words);
        const uint32_t num_data_words = static_cast<uint32_t>(data_words.Length());

        // Build the packed header:
        //   (sample_index << 28) | (type << 24) | (line << 10) | var_id
        const uint32_t static_bits =
            (static_cast<uint32_t>(dt) << L::kTypeShift) |
            ((line & L::kLineMask) << L::kLineShift) |
            (variable_id & L::kVariableIdMask);
        auto* si = b.Load(sample_index_var)->Result();
        auto* si_masked = b.And(si, u32(L::kSampleIndexMask))->Result();
        auto* si_shifted = b.ShiftLeft(si_masked, u32(L::kSampleIndexShift))->Result();
        auto* packed_header = b.Or(si_shifted, u32(static_bits))->Result();

        // Reserve 1 (header) + num_data_words entries in the records array.
        auto* cursor_ptr = b.Access(
            ty.ptr(core::AddressSpace::kStorage, ty.atomic<u32>(), core::Access::kReadWrite),
            debug_buffer_var, u32(kCursorMemberIndex));
        auto* slot = b.Call(ty.u32(), core::BuiltinFn::kAtomicAdd, cursor_ptr,
                            u32(1u + num_data_words))
                         ->Result();

        // records[slot] = packed_header
        auto* hdr_ptr = b.Access(
            ty.ptr(core::AddressSpace::kStorage, ty.u32(), core::Access::kReadWrite),
            debug_buffer_var, u32(kRecordsMemberIndex), slot);
        b.Store(hdr_ptr, packed_header);

        // records[slot + 1 + i] = data_words[i]
        for (uint32_t i = 0; i < num_data_words; i++) {
            auto* data_idx = b.Add(slot, u32(1u + i))->Result();
            auto* data_ptr = b.Access(
                ty.ptr(core::AddressSpace::kStorage, ty.u32(), core::Access::kReadWrite),
                debug_buffer_var, u32(kRecordsMemberIndex), data_idx);
            b.Store(data_ptr, data_words[i]);
        }
    }

    /// Emit a zero-data-word line-marker record into the builder's current
    /// insertion point. Header encodes type=kLineMarker, line=<line>.
    /// @param line the 13-bit source line number
    void EmitLineMarkerRecord(uint32_t line) {
        using L = ShaderVariableInstrumentationIdLayout;
        using DT = ShaderVariableInstrumentationDataType;

        const uint32_t static_bits =
            (static_cast<uint32_t>(DT::kLineMarker) << L::kTypeShift) |
            ((line & L::kLineMask) << L::kLineShift);
        auto* si = b.Load(sample_index_var)->Result();
        auto* si_masked = b.And(si, u32(L::kSampleIndexMask))->Result();
        auto* si_shifted = b.ShiftLeft(si_masked, u32(L::kSampleIndexShift))->Result();
        auto* packed_header = b.Or(si_shifted, u32(static_bits))->Result();

        // Reserve exactly 1 entry (header only).
        auto* cursor_ptr = b.Access(
            ty.ptr(core::AddressSpace::kStorage, ty.atomic<u32>(), core::Access::kReadWrite),
            debug_buffer_var, u32(kCursorMemberIndex));
        auto* slot =
            b.Call(ty.u32(), core::BuiltinFn::kAtomicAdd, cursor_ptr, 1_u)->Result();

        auto* hdr_ptr = b.Access(
            ty.ptr(core::AddressSpace::kStorage, ty.u32(), core::Access::kReadWrite),
            debug_buffer_var, u32(kRecordsMemberIndex), slot);
        b.Store(hdr_ptr, packed_header);
    }

    /// Emit a line marker before @p inst, optionally gated by the match flag.
    void EmitLineMarkerBefore(Instruction* inst, uint32_t line) {
        b.InsertBefore(inst, [&] {
            if (match_var == nullptr) {
                EmitLineMarkerRecord(line);
                return;
            }
            auto* cond = b.Load(match_var)->Result();
            auto* ifelse = b.If(cond);
            b.Append(ifelse->True(), [&] {
                EmitLineMarkerRecord(line);
                b.ExitIf(ifelse);
            });
        });
    }

    /// @returns true if the given instruction is a control-flow instruction
    /// that should get a line marker emitted before it.
    static bool IsMarkerTarget(Instruction* inst) {
        return inst->IsAnyOf<If, Loop, Switch, UserCall, Return>();
    }

    /// Emit the instrumentation sequence for a single store, optionally gated
    /// by the per-invocation match flag.
    /// @param store the store being instrumented
    /// @param variable_id the 10-bit variable id
    /// @param line the 16-bit source line number
    void EmitInstrumentation(Store* store,
                             uint32_t variable_id,
                             uint32_t line,
                             ShaderVariableInstrumentationDataType dt) {
        b.InsertAfter(store, [&] {
            if (match_var == nullptr) {
                EmitAppendRecord(store, variable_id, line, dt);
                return;
            }
            auto* cond = b.Load(match_var)->Result();
            auto* ifelse = b.If(cond);
            b.Append(ifelse->True(), [&] {
                EmitAppendRecord(store, variable_id, line, dt);
                b.ExitIf(ifelse);
            });
        });
    }

    /// Run the transform.
    ShaderVariableInstrumentationResult Run() {
        using L = ShaderVariableInstrumentationIdLayout;

        // First pass: collect all candidate stores — and, if line markers
        // are enabled, all candidate control-flow instructions — before
        // mutating the module. This avoids instrumenting instructions we
        // insert ourselves and keeps the iterator stable while we mutate.
        std::vector<Store*> candidates;
        std::vector<Instruction*> marker_candidates;
        for (auto* inst : ir.Instructions()) {
            if (auto* store = inst->As<Store>()) {
                if (ShouldInstrument(store)) {
                    candidates.push_back(store);
                }
            } else if (config.emit_line_markers && IsMarkerTarget(inst)) {
                marker_candidates.push_back(inst);
            }
        }

        if (candidates.empty() && marker_candidates.empty()) {
            return {};
        }

        // Create the debug buffer and the sample-index private var. It is
        // important that these happen *after* we have finished the candidate
        // scan, so that the Vars and Stores we introduce are never candidates.
        EnsureDebugBuffer();
        EnsureSampleIndexVar();
        if (config.target_fragment_coord.has_value()) {
            EnsureMatchVar();
        }
        SetupAllFragmentEntryPoints();

        // Second pass: instrument each candidate. Variable ids are assigned
        // per unique Var (not per store), so multiple stores to the same
        // variable share one id.
        for (auto* store : candidates) {
            auto* var = OriginatingVar(store->To());
            TINT_IR_ASSERT(ir, var != nullptr);

            // Get-or-create a variable_id for this Var.
            auto variable_id = var_id_map.GetOrAdd(var, [&]() {
                const uint32_t id =
                    static_cast<uint32_t>(variables.size()) & L::kVariableIdMask;

                auto dt = DataTypeOf(store->From()->Type());

                // Declaration source from the Var instruction itself.
                auto decl_src = ir.SourceOf(var);
                const uint32_t decl_line =
                    decl_src.range.begin.line <= L::kLineMask
                        ? static_cast<uint32_t>(decl_src.range.begin.line)
                        : L::kLineMask;

                ShaderVariableInstrumentationVariableInfo info;
                info.variable_id = id;
                info.data_type = dt;
                auto sym = ir.NameOf(var->Result());
                info.name = sym.IsValid() ? sym.Name() : "";
                info.declaration_line = decl_line;
                info.declaration_source = decl_src;
                variables.push_back(std::move(info));

                return id;
            });

            // The line in the packed header is the line of the *store*
            // instruction (where the update happened), not the declaration.
            auto store_src = ir.SourceOf(store);
            const uint32_t store_line =
                store_src.range.begin.line <= L::kLineMask
                    ? static_cast<uint32_t>(store_src.range.begin.line)
                    : L::kLineMask;

            auto dt = DataTypeOf(store->From()->Type());
            EmitInstrumentation(store, variable_id, store_line, dt);
        }

        // Third pass: emit a line-marker record before each collected
        // control-flow instruction. Instructions with no source info are
        // skipped.
        for (auto* inst : marker_candidates) {
            auto src = ir.SourceOf(inst);
            if (src.range.begin.line == 0) {
                continue;
            }
            const uint32_t line = src.range.begin.line <= L::kLineMask
                                      ? static_cast<uint32_t>(src.range.begin.line)
                                      : L::kLineMask;
            EmitLineMarkerBefore(inst, line);
        }

        ShaderVariableInstrumentationResult result;
        result.variables = std::move(variables);
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
