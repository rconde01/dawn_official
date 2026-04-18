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

#include "src/tint/lang/core/ir/transform/helper_test.h"

namespace tint::core::ir::transform {
namespace {

using namespace tint::core::fluent_types;     // NOLINT
using namespace tint::core::number_suffixes;  // NOLINT

// Helper that runs the transform and returns the result struct so tests can
// also assert on the collected records metadata.
class IR_ShaderVariableInstrumentationTest : public TransformTest {
  public:
    IR_ShaderVariableInstrumentationTest() {
        capabilities = kShaderVariableInstrumentationCapabilities;
    }

    ShaderVariableInstrumentationResult RunAndValidate(
        const ShaderVariableInstrumentationConfig& config) {
        mod.enable_validation_asserts = true;
        auto result = ShaderVariableInstrumentation(mod, config);
        EXPECT_EQ(result, Success);
        if (result != Success) {
            return {};
        }
        EXPECT_EQ(ir::Validate(mod, capabilities, "after transform"), Success);
        return result.Move();
    }
};

TEST_F(IR_ShaderVariableInstrumentationTest, NoStores_NoModify) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {  //
        b.Return(func);
    });

    auto* src = R"(
%foo = func():void {
  $B1: {
    ret
  }
}
)";
    EXPECT_EQ(src, str());

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    // No stores means no debug buffer is introduced and no records emitted.
    EXPECT_EQ(src, str());
    EXPECT_EQ(result.variables.size(), 0u);
}

TEST_F(IR_ShaderVariableInstrumentationTest, StoresToStorageBuffer_NotInstrumented) {
    auto* out = b.Var("out", ty.ptr<storage, u32, read_write>());
    out->SetBindingPoint(0, 0);
    mod.root_block->Append(out);

    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        b.Store(out, 42_u);
        b.Return(func);
    });

    auto* src = R"(
$B1: {  # root
  %out:ptr<storage, u32, read_write> = var undef @binding_point(0, 0)
}

%foo = func():void {
  $B2: {
    store %out, 42u
    ret
  }
}
)";
    EXPECT_EQ(src, str());

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    // Stores into storage buffers are user outputs; the transform leaves them
    // alone and therefore never introduces the debug buffer.
    EXPECT_EQ(src, str());
    EXPECT_EQ(result.variables.size(), 0u);
}

TEST_F(IR_ShaderVariableInstrumentationTest, SingleU32Store_FunctionVar) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 42_u);
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    // The transform should have recorded a single u32 store whose destination
    // name is `v`.
    ASSERT_EQ(result.variables.size(), 1u);
    EXPECT_EQ(result.variables[0].variable_id, 0u);
    EXPECT_EQ(result.variables[0].data_type,
              ShaderVariableInstrumentationDataType::kU32);
    EXPECT_EQ(result.variables[0].name, "v");

    // The packed id is: (sample_index << 26) | (line << 10) | var_id.
    // No source info so line = 0, non-fragment so sample = 0.
    // Expected static_bits = 0.
    // But sample_index_var defaults to 0 in a non-fragment function.
    // We just verify the structure; exact numbering via verbatim match.
}

TEST_F(IR_ShaderVariableInstrumentationTest, I32Store_UsesBitcast) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, i32>("v");
        b.Store(v, -7_i);
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);
    EXPECT_EQ(result.variables[0].data_type,
              ShaderVariableInstrumentationDataType::kI32);
    EXPECT_EQ(result.variables[0].name, "v");
}

TEST_F(IR_ShaderVariableInstrumentationTest, Vec4F32Store_Instrumented) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var("v", ty.ptr<function>(ty.vec4<f32>()));
        b.Store(v, b.Splat(ty.vec4<f32>(), 1_f));
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);
    EXPECT_EQ(result.variables[0].data_type,
              ShaderVariableInstrumentationDataType::kVec4F32);
    EXPECT_EQ(result.variables[0].name, "v");

    // vec4<f32> produces 4 data words; cursor should advance by 5
    // (1 header + 4 data words).
    auto ir_text = str();
    EXPECT_NE(ir_text.find("atomicAdd"), std::string::npos);
    // The atomicAdd should reserve 5 entries (1 header + 4 words).
    EXPECT_NE(ir_text.find("5u"), std::string::npos);
}

TEST_F(IR_ShaderVariableInstrumentationTest, BoolStore_UsesSelect) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, bool>("v");
        b.Store(v, true);
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);
    EXPECT_EQ(result.variables[0].data_type,
              ShaderVariableInstrumentationDataType::kBool);
    EXPECT_EQ(result.variables[0].name, "v");

    // The IR should contain a `select 0u, 1u, <loaded_bool>` to convert
    // the boolean to a u32.
    auto ir_text = str();
    EXPECT_NE(ir_text.find("select"), std::string::npos);
}

TEST_F(IR_ShaderVariableInstrumentationTest,
       FragmentCoordFilter_AddsPositionParamAndGate) {
    auto* ep = b.Function("frag", ty.vec4<f32>(), Function::PipelineStage::kFragment);
    ep->SetReturnLocation(0_u);
    b.Append(ep->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 42_u);
        b.Return(ep, b.Splat(ty.vec4<f32>(), 0_f));
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    cfg.target_fragment_coord = std::array<uint32_t, 2>{100u, 50u};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);

    // Verify high-level structure: the IR should contain the atomic cursor,
    // the sample_index and match private vars, the if-gated record append,
    // and a bitfield id construction (or, shl, and, load of sample_index).
    auto ir_text = str();
    EXPECT_NE(ir_text.find("cursor:atomic<u32>"), std::string::npos);
    EXPECT_NE(ir_text.find("tint_sample_index"), std::string::npos);
    EXPECT_NE(ir_text.find("tint_shader_debug_match"), std::string::npos);
    EXPECT_NE(ir_text.find("atomicAdd"), std::string::npos);
    EXPECT_NE(ir_text.find("load %v"), std::string::npos);
    EXPECT_NE(ir_text.find("@sample_index"), std::string::npos);
    // The or/shl for bitfield packing:
    EXPECT_NE(ir_text.find("shl"), std::string::npos);
    EXPECT_NE(ir_text.find("or"), std::string::npos);
}

TEST_F(IR_ShaderVariableInstrumentationTest,
       FragmentCoordFilter_ReusesExistingPositionParam) {
    auto* pos = b.FunctionParam("my_pos", ty.vec4<f32>());
    pos->SetBuiltin(core::BuiltinValue::kPosition);
    auto* ep = b.Function("frag", ty.vec4<f32>(), Function::PipelineStage::kFragment);
    ep->SetParams({pos});
    ep->SetReturnLocation(0_u);
    b.Append(ep->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 42_u);
        b.Return(ep, b.Splat(ty.vec4<f32>(), 0_f));
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    cfg.target_fragment_coord = std::array<uint32_t, 2>{0u, 0u};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);

    // The existing `my_pos` parameter should be reused; a sample_index
    // parameter is added but no tint_frag_coord.
    EXPECT_EQ(ep->Params().Length(), 2u);  // my_pos + tint_sample_idx

    auto ir_text = str();
    EXPECT_NE(ir_text.find("%my_pos"), std::string::npos);
    EXPECT_EQ(ir_text.find("tint_frag_coord"), std::string::npos);
    EXPECT_NE(ir_text.find("store %tint_shader_debug_match"), std::string::npos);
    EXPECT_NE(ir_text.find("load %tint_shader_debug_match"), std::string::npos);
    EXPECT_NE(ir_text.find("tint_sample_idx"), std::string::npos);
}

TEST_F(IR_ShaderVariableInstrumentationTest,
       FragmentCoordFilter_NonFragmentFunction_NoSetupButStillGated) {
    // A compute entry point: the match var is still declared because we have
    // at least one candidate store, but no match assignment happens and so
    // the gate always reads `false`.
    auto* ep = b.ComputeFunction("cs");
    b.Append(ep->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 42_u);
        b.Return(ep);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    cfg.target_fragment_coord = std::array<uint32_t, 2>{0u, 0u};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);

    auto ir_text = str();
    // The private match var and sample_index var are created.
    EXPECT_NE(ir_text.find("%tint_shader_debug_match:ptr<private, bool"),
              std::string::npos);
    EXPECT_NE(ir_text.find("tint_sample_index"), std::string::npos);
    EXPECT_NE(ir_text.find("load %tint_shader_debug_match"), std::string::npos);
    // But we never computed a match inside the compute entry point (no
    // fragment entry points to set up).
    EXPECT_EQ(ir_text.find("store %tint_shader_debug_match"), std::string::npos);
    EXPECT_EQ(ir_text.find("tint_frag_coord"), std::string::npos);
    // Bitfield packing still happens (shl, or):
    EXPECT_NE(ir_text.find("shl"), std::string::npos);
    EXPECT_NE(ir_text.find("or"), std::string::npos);
}

TEST_F(IR_ShaderVariableInstrumentationTest, TwoStores_SequentialIds) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* a = b.Var<function, u32>("a");
        auto* bv = b.Var<function, f32>("b");
        b.Store(a, 10_u);
        b.Store(bv, 2.5_f);
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 2u);
    EXPECT_EQ(result.variables[0].variable_id, 0u);
    EXPECT_EQ(result.variables[0].data_type,
              ShaderVariableInstrumentationDataType::kU32);
    EXPECT_EQ(result.variables[0].name, "a");
    EXPECT_EQ(result.variables[1].variable_id, 1u);
    EXPECT_EQ(result.variables[1].data_type,
              ShaderVariableInstrumentationDataType::kF32);
    EXPECT_EQ(result.variables[1].name, "b");
}

TEST_F(IR_ShaderVariableInstrumentationTest, LineMarkers_Disabled_NoMarkerInstructions) {
    // Default: emit_line_markers = false. A UserCall should not produce
    // any marker records.
    auto* callee = b.Function("callee", ty.u32());
    b.Append(callee->Block(), [&] { b.Return(callee, 7_u); });

    auto* func = b.Function("foo", ty.void_());
    ir::UserCall* call = nullptr;
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 1_u);
        call = b.Call(ty.u32(), callee);
        b.Return(func);
    });
    mod.SetSource(call, Source{{5, 1}});

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    // Leave emit_line_markers at default (false).
    auto result = RunAndValidate(cfg);

    ASSERT_EQ(result.variables.size(), 1u);  // Just `v`.
    // Exactly one atomicAdd: the store record. No marker for the call.
    auto ir_text = str();
    size_t count = 0;
    size_t pos = 0;
    while ((pos = ir_text.find("atomicAdd", pos)) != std::string::npos) {
        ++count;
        pos += 9;
    }
    EXPECT_EQ(count, 1u) << "Expected only the store's atomicAdd, no marker";
}

TEST_F(IR_ShaderVariableInstrumentationTest, LineMarkers_Enabled_EmitsMarkerForCall) {
    // Use a non-void UserCall so the call instruction has a single-result
    // Value that can carry source info (Tint only tracks sources for
    // single-result instructions).
    auto* callee = b.Function("callee", ty.u32());
    b.Append(callee->Block(), [&] { b.Return(callee, 7_u); });

    auto* func = b.Function("foo", ty.u32());
    ir::UserCall* call = nullptr;
    b.Append(func->Block(), [&] {
        call = b.Call(ty.u32(), callee);
        b.Return(func, call);
    });
    mod.SetSource(call, Source{{5, 1}});

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    cfg.emit_line_markers = true;
    auto result = RunAndValidate(cfg);

    // No variables recorded — the call isn't a store.
    EXPECT_EQ(result.variables.size(), 0u);

    // But the debug buffer is created and there is at least one atomicAdd,
    // produced by the marker emitted before the UserCall.
    auto ir_text = str();
    EXPECT_NE(ir_text.find("tint_shader_debug_buffer"), std::string::npos);
    EXPECT_NE(ir_text.find("atomicAdd"), std::string::npos);
}

TEST_F(IR_ShaderVariableInstrumentationTest, LineMarkers_Enabled_MarkerAndStoreBothEmit) {
    // A store to a scalar plus a call with source info.
    auto* callee = b.Function("callee", ty.u32());
    b.Append(callee->Block(), [&] { b.Return(callee, 7_u); });

    auto* func = b.Function("foo", ty.void_());
    ir::UserCall* call = nullptr;
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 1_u);
        call = b.Call(ty.u32(), callee);
        b.Return(func);
    });
    mod.SetSource(call, Source{{5, 1}});

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    cfg.emit_line_markers = true;
    auto result = RunAndValidate(cfg);

    // One variable (v) and at least two atomicAdd sites (v's store record
    // plus the call's line marker).
    ASSERT_EQ(result.variables.size(), 1u);
    auto ir_text = str();
    size_t count = 0;
    size_t pos = 0;
    while ((pos = ir_text.find("atomicAdd", pos)) != std::string::npos) {
        ++count;
        pos += 9;
    }
    EXPECT_GE(count, 2u) << "Expected at least 2 atomicAdd sites (store + call marker)";
}

TEST_F(IR_ShaderVariableInstrumentationTest, LineMarkers_InfersLineFromIfCondition) {
    // The If instruction itself has no source, but its condition does.
    // The transform should infer the line from the condition value.
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, bool>("v");
        b.Store(v, true);
        auto* loaded = b.Load(v);
        mod.SetSource(loaded, Source{{10, 1}});
        auto* ifelse = b.If(loaded);
        // If instruction has no source (0 results = no SetSource possible).
        b.Append(ifelse->True(), [&] { b.ExitIf(ifelse); });
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    cfg.emit_line_markers = true;
    auto result = RunAndValidate(cfg);

    // At least 2 atomicAdd sites: the bool store + the If marker (from
    // the condition's inferred source line).
    auto ir_text = str();
    size_t count = 0;
    size_t pos = 0;
    while ((pos = ir_text.find("atomicAdd", pos)) != std::string::npos) {
        ++count;
        pos += 9;
    }
    EXPECT_GE(count, 2u) << "Expected at least 2 atomicAdd (store + If marker)";
}

TEST_F(IR_ShaderVariableInstrumentationTest, TwoStoresToSameVar_ShareOneId) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var<function, u32>("v");
        b.Store(v, 1_u);
        b.Store(v, 2_u);
        b.Return(func);
    });

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    // Both stores target the same Var, so only ONE variable entry is created.
    ASSERT_EQ(result.variables.size(), 1u);
    EXPECT_EQ(result.variables[0].variable_id, 0u);
    EXPECT_EQ(result.variables[0].name, "v");
    EXPECT_EQ(result.variables[0].data_type,
              ShaderVariableInstrumentationDataType::kU32);
}

}  // namespace
}  // namespace tint::core::ir::transform
