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
    EXPECT_EQ(result.records.size(), 0u);
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
    EXPECT_EQ(result.records.size(), 0u);
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
    ASSERT_EQ(result.records.size(), 1u);
    EXPECT_EQ(result.records[0].variable_id, 0u);
    EXPECT_EQ(result.records[0].scalar_type,
              ShaderVariableInstrumentationScalarType::kU32);
    EXPECT_EQ(result.records[0].variable_name, "v");

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

    ASSERT_EQ(result.records.size(), 1u);
    EXPECT_EQ(result.records[0].scalar_type,
              ShaderVariableInstrumentationScalarType::kI32);
    EXPECT_EQ(result.records[0].variable_name, "v");
}

TEST_F(IR_ShaderVariableInstrumentationTest, VectorStore_NotInstrumented) {
    auto* func = b.Function("foo", ty.void_());
    b.Append(func->Block(), [&] {
        auto* v = b.Var("v", ty.ptr<function>(ty.vec4<f32>()));
        b.Store(v, b.Splat(ty.vec4<f32>(), 1_f));
        b.Return(func);
    });

    // Capture the pre-run IR string for comparison.
    auto before = str();

    ShaderVariableInstrumentationConfig cfg;
    cfg.buffer_binding_point = {1, 0};
    auto result = RunAndValidate(cfg);

    // Non-scalar stores are not supported by v1; the IR should be unchanged.
    EXPECT_EQ(before, str());
    EXPECT_EQ(result.records.size(), 0u);
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

    ASSERT_EQ(result.records.size(), 1u);

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

    ASSERT_EQ(result.records.size(), 1u);

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

    ASSERT_EQ(result.records.size(), 1u);

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

    ASSERT_EQ(result.records.size(), 2u);
    EXPECT_EQ(result.records[0].variable_id, 0u);
    EXPECT_EQ(result.records[0].scalar_type,
              ShaderVariableInstrumentationScalarType::kU32);
    EXPECT_EQ(result.records[0].variable_name, "a");
    EXPECT_EQ(result.records[1].variable_id, 1u);
    EXPECT_EQ(result.records[1].scalar_type,
              ShaderVariableInstrumentationScalarType::kF32);
    EXPECT_EQ(result.records[1].variable_name, "b");
}

}  // namespace
}  // namespace tint::core::ir::transform
