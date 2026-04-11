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
    EXPECT_EQ(result.records[0].id, 0u);
    EXPECT_EQ(result.records[0].scalar_type,
              ShaderVariableInstrumentationScalarType::kU32);
    EXPECT_EQ(result.records[0].variable_name, "v");

    auto* expect = R"(
tint_shader_debug_buffer = struct @align(4) {
  cursor:atomic<u32> @offset(0)
  records:array<u32> @offset(4)
}

$B1: {  # root
  %tint_shader_debug_buffer:ptr<storage, tint_shader_debug_buffer, read_write> = var undef @binding_point(1, 0)
}

%foo = func():void {
  $B2: {
    %v:ptr<function, u32, read_write> = var undef
    store %v, 42u
    %4:ptr<storage, atomic<u32>, read_write> = access %tint_shader_debug_buffer, 0u
    %5:u32 = atomicAdd %4, 1u
    %6:u32 = mul %5, 2u
    %7:u32 = add %6, 1u
    %8:ptr<storage, u32, read_write> = access %tint_shader_debug_buffer, 1u, %6
    store %8, 0u
    %9:ptr<storage, u32, read_write> = access %tint_shader_debug_buffer, 1u, %7
    store %9, 42u
    ret
  }
}
)";
    EXPECT_EQ(expect, str());
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

    auto* expect = R"(
tint_shader_debug_buffer = struct @align(4) {
  cursor:atomic<u32> @offset(0)
  records:array<u32> @offset(4)
}

$B1: {  # root
  %tint_shader_debug_buffer:ptr<storage, tint_shader_debug_buffer, read_write> = var undef @binding_point(1, 0)
}

%foo = func():void {
  $B2: {
    %v:ptr<function, i32, read_write> = var undef
    store %v, -7i
    %4:ptr<storage, atomic<u32>, read_write> = access %tint_shader_debug_buffer, 0u
    %5:u32 = atomicAdd %4, 1u
    %6:u32 = mul %5, 2u
    %7:u32 = add %6, 1u
    %8:ptr<storage, u32, read_write> = access %tint_shader_debug_buffer, 1u, %6
    store %8, 0u
    %9:u32 = bitcast<u32> -7i
    %10:ptr<storage, u32, read_write> = access %tint_shader_debug_buffer, 1u, %7
    store %10, %9
    ret
  }
}
)";
    EXPECT_EQ(expect, str());
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

    auto* expect = R"(
tint_shader_debug_buffer = struct @align(4) {
  cursor:atomic<u32> @offset(0)
  records:array<u32> @offset(4)
}

$B1: {  # root
  %tint_shader_debug_buffer:ptr<storage, tint_shader_debug_buffer, read_write> = var undef @binding_point(1, 0)
  %tint_shader_debug_match:ptr<private, bool, read_write> = var false
}

%frag = @fragment func(%tint_frag_coord:vec4<f32> [@position]):vec4<f32> [@location(0)] {
  $B2: {
    %5:f32 = access %tint_frag_coord, 0u
    %6:f32 = access %tint_frag_coord, 1u
    %7:u32 = convert %5
    %8:u32 = convert %6
    %9:bool = eq %7, 100u
    %10:bool = eq %8, 50u
    %11:bool = and %9, %10
    store %tint_shader_debug_match, %11
    %v:ptr<function, u32, read_write> = var undef
    store %v, 42u
    %13:bool = load %tint_shader_debug_match
    if %13 [t: $B3] {  # if_1
      $B3: {  # true
        %14:ptr<storage, atomic<u32>, read_write> = access %tint_shader_debug_buffer, 0u
        %15:u32 = atomicAdd %14, 1u
        %16:u32 = mul %15, 2u
        %17:u32 = add %16, 1u
        %18:ptr<storage, u32, read_write> = access %tint_shader_debug_buffer, 1u, %16
        store %18, 0u
        %19:ptr<storage, u32, read_write> = access %tint_shader_debug_buffer, 1u, %17
        store %19, 42u
        exit_if  # if_1
      }
    }
    ret vec4<f32>(0.0f)
  }
}
)";
    EXPECT_EQ(expect, str());
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

    // The existing `my_pos` parameter should be reused; no new parameter is
    // appended.
    EXPECT_EQ(ep->Params().Length(), 1u);

    // The entry point body now opens with the match computation using the
    // existing parameter.
    auto ir_text = str();
    EXPECT_NE(ir_text.find("%my_pos"), std::string::npos);
    EXPECT_EQ(ir_text.find("tint_frag_coord"), std::string::npos);
    EXPECT_NE(ir_text.find("store %tint_shader_debug_match"), std::string::npos);
    EXPECT_NE(ir_text.find("load %tint_shader_debug_match"), std::string::npos);
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
    // The private match var is still created and the store is gated.
    EXPECT_NE(ir_text.find("%tint_shader_debug_match:ptr<private, bool"),
              std::string::npos);
    EXPECT_NE(ir_text.find("load %tint_shader_debug_match"), std::string::npos);
    // But we never computed a match inside the compute entry point.
    EXPECT_EQ(ir_text.find("store %tint_shader_debug_match"), std::string::npos);
    EXPECT_EQ(ir_text.find("tint_frag_coord"), std::string::npos);
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
    EXPECT_EQ(result.records[0].id, 0u);
    EXPECT_EQ(result.records[0].scalar_type,
              ShaderVariableInstrumentationScalarType::kU32);
    EXPECT_EQ(result.records[0].variable_name, "a");
    EXPECT_EQ(result.records[1].id, 1u);
    EXPECT_EQ(result.records[1].scalar_type,
              ShaderVariableInstrumentationScalarType::kF32);
    EXPECT_EQ(result.records[1].variable_name, "b");
}

}  // namespace
}  // namespace tint::core::ir::transform
