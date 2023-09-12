; ModuleID = ""
target triple = "wasm32-unknown-unknown"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"

define void @"main"(float* %".1")
{
buffers:
  %"buf_0" = load float*, float** @"buf_0"
  %"buf_1" = load float*, float** @"buf_1"
  %"buf_2" = getelementptr [10 x float], [10 x float]* @"buf_2", i32 0, i32 0
  %"buf_3" = getelementptr [100 x float], [100 x float]* @"buf_3", i32 0, i32 0
  %"buf_4" = getelementptr [100 x float], [100 x float]* @"buf_4", i32 0, i32 0
  %"buf_5" = getelementptr [100 x float], [100 x float]* @"buf_5", i32 0, i32 0
  %"buf_6" = getelementptr [10 x float], [10 x float]* @"buf_6", i32 0, i32 0
  %"buf_7" = getelementptr [10 x float], [10 x float]* @"buf_7", i32 0, i32 0
  %"buf_8" = getelementptr [100 x float], [100 x float]* @"buf_8", i32 0, i32 0
  %"buf_9" = getelementptr [100 x float], [100 x float]* @"buf_9", i32 0, i32 0
  br label %"main"
main:
  call void @"ShapeOPS.MAX_10_10_1"(float* %"buf_1", float* %"buf_2")
  call void @"ReshapeOPS.CAST_10_1_10_10"(float* %"buf_2", float* %"buf_3")
  call void @"BinaryOPS.SUB_10_10"(float* %"buf_1", float* %"buf_3", float* %"buf_4")
  call void @"UnaryOPS.EXP_10_10"(float* %"buf_4", float* %"buf_5")
  call void @"ShapeOPS.SUM_10_10_1"(float* %"buf_5", float* %"buf_6")
  call void @"UnaryOPS.LOG_10_1"(float* %"buf_6", float* %"buf_7")
  call void @"ReshapeOPS.CAST_10_1_10_10"(float* %"buf_7", float* %"buf_8")
  call void @"BinaryOPS.SUB_10_10"(float* %"buf_4", float* %"buf_8", float* %".1")
  br label %"exit"
exit:
  ret void
}

@"raw_buf_0" = dso_local global [400 x i8] [i8 203, i8 25, i8 79, i8 61, i8 159, i8 249, i8 255, i8 62, i8 227, i8 243, i8 126, i8 191, i8 172, i8 143, i8 49, i8 63, i8 158, i8 43, i8 214, i8 190, i8 109, i8 211, i8 202, i8 191, i8 28, i8 208, i8 37, i8 191, i8 57, i8 60, i8 25, i8 63, i8 173, i8 28, i8 170, i8 62, i8 132, i8 224, i8 146, i8 191, i8 35, i8 97, i8 30, i8 63, i8 126, i8 50, i8 180, i8 189, i8 23, i8 163, i8 217, i8 62, i8 22, i8 29, i8 170, i8 62, i8 142, i8 18, i8 148, i8 191, i8 230, i8 181, i8 179, i8 62, i8 247, i8 92, i8 27, i8 191, i8 107, i8 3, i8 198, i8 63, i8 234, i8 44, i8 57, i8 63, i8 166, i8 248, i8 60, i8 61, i8 87, i8 165, i8 123, i8 191, i8 221, i8 244, i8 94, i8 61, i8 249, i8 186, i8 35, i8 62, i8 208, i8 190, i8 154, i8 191, i8 137, i8 75, i8 14, i8 64, i8 16, i8 225, i8 201, i8 62, i8 45, i8 159, i8 216, i8 63, i8 161, i8 112, i8 142, i8 191, i8 45, i8 96, i8 209, i8 63, i8 31, i8 52, i8 174, i8 191, i8 188, i8 182, i8 38, i8 191, i8 23, i8 222, i8 10, i8 63, i8 51, i8 162, i8 68, i8 61, i8 174, i8 234, i8 22, i8 192, i8 199, i8 131, i8 141, i8 191, i8 113, i8 124, i8 86, i8 63, i8 173, i8 159, i8 5, i8 64, i8 4, i8 51, i8 106, i8 63, i8 135, i8 106, i8 141, i8 190, i8 52, i8 232, i8 75, i8 63, i8 254, i8 103, i8 146, i8 191, i8 26, i8 138, i8 2, i8 63, i8 148, i8 121, i8 172, i8 191, i8 28, i8 91, i8 25, i8 188, i8 112, i8 215, i8 5, i8 190, i8 140, i8 85, i8 77, i8 63, i8 24, i8 30, i8 155, i8 190, i8 57, i8 219, i8 153, i8 63, i8 152, i8 119, i8 73, i8 190, i8 191, i8 38, i8 86, i8 63, i8 196, i8 94, i8 73, i8 63, i8 210, i8 161, i8 235, i8 191, i8 101, i8 203, i8 25, i8 61, i8 75, i8 41, i8 19, i8 61, i8 128, i8 91, i8 71, i8 191, i8 113, i8 183, i8 55, i8 62, i8 243, i8 78, i8 186, i8 191, i8 40, i8 98, i8 14, i8 63, i8 222, i8 128, i8 2, i8 63, i8 0, i8 212, i8 153, i8 62, i8 91, i8 128, i8 30, i8 64, i8 91, i8 102, i8 180, i8 62, i8 60, i8 46, i8 138, i8 61, i8 179, i8 117, i8 59, i8 191, i8 229, i8 34, i8 152, i8 62, i8 1, i8 55, i8 118, i8 191, i8 244, i8 202, i8 162, i8 63, i8 8, i8 204, i8 37, i8 191, i8 215, i8 69, i8 34, i8 62, i8 10, i8 187, i8 254, i8 63, i8 25, i8 4, i8 149, i8 63, i8 232, i8 123, i8 120, i8 62, i8 57, i8 161, i8 176, i8 63, i8 243, i8 120, i8 95, i8 189, i8 116, i8 148, i8 75, i8 63, i8 141, i8 98, i8 156, i8 60, i8 203, i8 202, i8 103, i8 191, i8 134, i8 76, i8 220, i8 62, i8 58, i8 69, i8 111, i8 63, i8 68, i8 52, i8 177, i8 190, i8 125, i8 110, i8 140, i8 191, i8 220, i8 55, i8 7, i8 191, i8 61, i8 78, i8 24, i8 192, i8 41, i8 145, i8 27, i8 191, i8 27, i8 163, i8 137, i8 191, i8 22, i8 111, i8 1, i8 64, i8 171, i8 155, i8 16, i8 191, i8 179, i8 126, i8 197, i8 191, i8 125, i8 239, i8 94, i8 63, i8 99, i8 106, i8 51, i8 190, i8 242, i8 19, i8 71, i8 61, i8 120, i8 44, i8 65, i8 62, i8 70, i8 86, i8 86, i8 62, i8 62, i8 183, i8 191, i8 190, i8 33, i8 103, i8 116, i8 63, i8 143, i8 243, i8 5, i8 63, i8 237, i8 219, i8 253, i8 190, i8 107, i8 148, i8 47, i8 190, i8 36, i8 194, i8 113, i8 191, i8 127, i8 205, i8 143, i8 62], align 16
@"buf_0" = dso_local global float* bitcast ([400 x i8]* @"raw_buf_0" to float*), align 8
@"raw_buf_1" = dso_local global [400 x i8] [i8 203, i8 25, i8 79, i8 61, i8 159, i8 249, i8 255, i8 62, i8 227, i8 243, i8 126, i8 191, i8 172, i8 143, i8 49, i8 63, i8 158, i8 43, i8 214, i8 190, i8 109, i8 211, i8 202, i8 191, i8 28, i8 208, i8 37, i8 191, i8 57, i8 60, i8 25, i8 63, i8 173, i8 28, i8 170, i8 62, i8 132, i8 224, i8 146, i8 191, i8 35, i8 97, i8 30, i8 63, i8 126, i8 50, i8 180, i8 189, i8 23, i8 163, i8 217, i8 62, i8 22, i8 29, i8 170, i8 62, i8 142, i8 18, i8 148, i8 191, i8 230, i8 181, i8 179, i8 62, i8 247, i8 92, i8 27, i8 191, i8 107, i8 3, i8 198, i8 63, i8 234, i8 44, i8 57, i8 63, i8 166, i8 248, i8 60, i8 61, i8 87, i8 165, i8 123, i8 191, i8 221, i8 244, i8 94, i8 61, i8 249, i8 186, i8 35, i8 62, i8 208, i8 190, i8 154, i8 191, i8 137, i8 75, i8 14, i8 64, i8 16, i8 225, i8 201, i8 62, i8 45, i8 159, i8 216, i8 63, i8 161, i8 112, i8 142, i8 191, i8 45, i8 96, i8 209, i8 63, i8 31, i8 52, i8 174, i8 191, i8 188, i8 182, i8 38, i8 191, i8 23, i8 222, i8 10, i8 63, i8 51, i8 162, i8 68, i8 61, i8 174, i8 234, i8 22, i8 192, i8 199, i8 131, i8 141, i8 191, i8 113, i8 124, i8 86, i8 63, i8 173, i8 159, i8 5, i8 64, i8 4, i8 51, i8 106, i8 63, i8 135, i8 106, i8 141, i8 190, i8 52, i8 232, i8 75, i8 63, i8 254, i8 103, i8 146, i8 191, i8 26, i8 138, i8 2, i8 63, i8 148, i8 121, i8 172, i8 191, i8 28, i8 91, i8 25, i8 188, i8 112, i8 215, i8 5, i8 190, i8 140, i8 85, i8 77, i8 63, i8 24, i8 30, i8 155, i8 190, i8 57, i8 219, i8 153, i8 63, i8 152, i8 119, i8 73, i8 190, i8 191, i8 38, i8 86, i8 63, i8 196, i8 94, i8 73, i8 63, i8 210, i8 161, i8 235, i8 191, i8 101, i8 203, i8 25, i8 61, i8 75, i8 41, i8 19, i8 61, i8 128, i8 91, i8 71, i8 191, i8 113, i8 183, i8 55, i8 62, i8 243, i8 78, i8 186, i8 191, i8 40, i8 98, i8 14, i8 63, i8 222, i8 128, i8 2, i8 63, i8 0, i8 212, i8 153, i8 62, i8 91, i8 128, i8 30, i8 64, i8 91, i8 102, i8 180, i8 62, i8 60, i8 46, i8 138, i8 61, i8 179, i8 117, i8 59, i8 191, i8 229, i8 34, i8 152, i8 62, i8 1, i8 55, i8 118, i8 191, i8 244, i8 202, i8 162, i8 63, i8 8, i8 204, i8 37, i8 191, i8 215, i8 69, i8 34, i8 62, i8 10, i8 187, i8 254, i8 63, i8 25, i8 4, i8 149, i8 63, i8 232, i8 123, i8 120, i8 62, i8 57, i8 161, i8 176, i8 63, i8 243, i8 120, i8 95, i8 189, i8 116, i8 148, i8 75, i8 63, i8 141, i8 98, i8 156, i8 60, i8 203, i8 202, i8 103, i8 191, i8 134, i8 76, i8 220, i8 62, i8 58, i8 69, i8 111, i8 63, i8 68, i8 52, i8 177, i8 190, i8 125, i8 110, i8 140, i8 191, i8 220, i8 55, i8 7, i8 191, i8 61, i8 78, i8 24, i8 192, i8 41, i8 145, i8 27, i8 191, i8 27, i8 163, i8 137, i8 191, i8 22, i8 111, i8 1, i8 64, i8 171, i8 155, i8 16, i8 191, i8 179, i8 126, i8 197, i8 191, i8 125, i8 239, i8 94, i8 63, i8 99, i8 106, i8 51, i8 190, i8 242, i8 19, i8 71, i8 61, i8 120, i8 44, i8 65, i8 62, i8 70, i8 86, i8 86, i8 62, i8 62, i8 183, i8 191, i8 190, i8 33, i8 103, i8 116, i8 63, i8 143, i8 243, i8 5, i8 63, i8 237, i8 219, i8 253, i8 190, i8 107, i8 148, i8 47, i8 190, i8 36, i8 194, i8 113, i8 191, i8 127, i8 205, i8 143, i8 62], align 16
@"buf_1" = dso_local global float* bitcast ([400 x i8]* @"raw_buf_1" to float*), align 8
@"buf_2" = dso_local global [10 x float] zeroinitializer, align 16
define void @"ShapeOPS.MAX_10_10_1"(float* %".1", float* %".2")
{
entry:
  br label %"localidx"
localidx:
  %".6" = phi  i32 [0, %"entry"], [%".68", %"localidx"]
  %".7" = mul i32 %".6", 10
  %".8" = add i32 %".7", 0
  %".9" = getelementptr inbounds float, float* %".1", i32 %".8"
  %".10" = load float, float* %".9"
  %".11" = mul i32 %".6", 10
  %".12" = add i32 %".11", 1
  %".13" = getelementptr inbounds float, float* %".1", i32 %".12"
  %".14" = load float, float* %".13"
  %".15" = mul i32 %".6", 10
  %".16" = add i32 %".15", 2
  %".17" = getelementptr inbounds float, float* %".1", i32 %".16"
  %".18" = load float, float* %".17"
  %".19" = mul i32 %".6", 10
  %".20" = add i32 %".19", 3
  %".21" = getelementptr inbounds float, float* %".1", i32 %".20"
  %".22" = load float, float* %".21"
  %".23" = mul i32 %".6", 10
  %".24" = add i32 %".23", 4
  %".25" = getelementptr inbounds float, float* %".1", i32 %".24"
  %".26" = load float, float* %".25"
  %".27" = mul i32 %".6", 10
  %".28" = add i32 %".27", 5
  %".29" = getelementptr inbounds float, float* %".1", i32 %".28"
  %".30" = load float, float* %".29"
  %".31" = mul i32 %".6", 10
  %".32" = add i32 %".31", 6
  %".33" = getelementptr inbounds float, float* %".1", i32 %".32"
  %".34" = load float, float* %".33"
  %".35" = mul i32 %".6", 10
  %".36" = add i32 %".35", 7
  %".37" = getelementptr inbounds float, float* %".1", i32 %".36"
  %".38" = load float, float* %".37"
  %".39" = mul i32 %".6", 10
  %".40" = add i32 %".39", 8
  %".41" = getelementptr inbounds float, float* %".1", i32 %".40"
  %".42" = load float, float* %".41"
  %".43" = mul i32 %".6", 10
  %".44" = add i32 %".43", 9
  %".45" = getelementptr inbounds float, float* %".1", i32 %".44"
  %".46" = load float, float* %".45"
  %".47" = fadd float %".10",              0x0
  %".48" = fcmp ugt float %".14", %".47"
  %".49" = select  i1 %".48", float %".14", float %".47"
  %".50" = fcmp ugt float %".18", %".49"
  %".51" = select  i1 %".50", float %".18", float %".49"
  %".52" = fcmp ugt float %".22", %".51"
  %".53" = select  i1 %".52", float %".22", float %".51"
  %".54" = fcmp ugt float %".26", %".53"
  %".55" = select  i1 %".54", float %".26", float %".53"
  %".56" = fcmp ugt float %".30", %".55"
  %".57" = select  i1 %".56", float %".30", float %".55"
  %".58" = fcmp ugt float %".34", %".57"
  %".59" = select  i1 %".58", float %".34", float %".57"
  %".60" = fcmp ugt float %".38", %".59"
  %".61" = select  i1 %".60", float %".38", float %".59"
  %".62" = fcmp ugt float %".42", %".61"
  %".63" = select  i1 %".62", float %".42", float %".61"
  %".64" = fcmp ugt float %".46", %".63"
  %".65" = select  i1 %".64", float %".46", float %".63"
  %".66" = getelementptr inbounds float, float* %".2", i32 %".6"
  store float %".65", float* %".66"
  %".68" = add i32 %".6", 1
  %".69" = icmp eq i32 %".68", 10
  br i1 %".69", label %"out", label %"localidx"
out:
  ret void
}

@"buf_3" = dso_local global [100 x float] zeroinitializer, align 16
define void @"ReshapeOPS.CAST_10_1_10_10"(float* %".1", float* %".2")
{
entry:
  br label %"globalidx"
globalidx:
  %".4" = phi  i32 [0, %"entry"], [%".17", %"globalidx_edit"]
  br label %"localidx"
localidx:
  %".5" = phi  i32 [0, %"globalidx"], [%".14", %"localidx"]
  %".8" = getelementptr float, float* %".1", i32 %".4"
  %".9" = load float, float* %".8"
  %".10" = mul i32 %".4", 10
  %".11" = add i32 %".10", %".5"
  %".12" = getelementptr float, float* %".2", i32 %".11"
  store float %".9", float* %".12"
  %".14" = add i32 %".5", 1
  %".15" = icmp eq i32 %".14", 10
  br i1 %".15", label %"globalidx_edit", label %"localidx"
globalidx_edit:
  %".17" = add i32 %".4", 1
  %".18" = icmp eq i32 %".17", 10
  br i1 %".18", label %"out", label %"globalidx"
out:
  ret void
}

@"buf_4" = dso_local global [100 x float] zeroinitializer, align 16
define void @"BinaryOPS.SUB_10_10"(float* %".1", float* %".2", float* %".3")
{
entry:
  br label %"loop"
loop:
  %".7" = phi  i32 [0, %"entry"], [%".15", %"loop"]
  %".8" = getelementptr float, float* %".1", i32 %".7"
  %".9" = load float, float* %".8"
  %".10" = getelementptr float, float* %".2", i32 %".7"
  %".11" = load float, float* %".10"
  %".12" = getelementptr float, float* %".3", i32 %".7"
  %".13" = fsub float %".9", %".11"
  store float %".13", float* %".12"
  %".15" = add i32 %".7", 1
  %".16" = icmp eq i32 %".15", 100
  br i1 %".16", label %"out", label %"loop"
out:
  ret void
}

@"buf_5" = dso_local global [100 x float] zeroinitializer, align 16
define void @"UnaryOPS.EXP_10_10"(float* %".1", float* %".2")
{
entry:
  br label %"loop"
loop:
  %".6" = phi  i32 [0, %"entry"], [%".12", %"loop"]
  %".7" = getelementptr float, float* %".1", i32 %".6"
  %".8" = load float, float* %".7"
  %".9" = getelementptr float, float* %".2", i32 %".6"
  %".10" = call float @"llvm.exp.f32"(float %".8")
  store float %".10", float* %".9"
  %".12" = add i32 %".6", 1
  %".13" = icmp eq i32 %".12", 100
  br i1 %".13", label %"out", label %"loop"
out:
  ret void
}

declare float @"llvm.exp.f32"(float %".1")

@"buf_6" = dso_local global [10 x float] zeroinitializer, align 16
define void @"ShapeOPS.SUM_10_10_1"(float* %".1", float* %".2")
{
entry:
  br label %"localidx"
localidx:
  %".6" = phi  i32 [0, %"entry"], [%".59", %"localidx"]
  %".7" = mul i32 %".6", 10
  %".8" = add i32 %".7", 0
  %".9" = getelementptr inbounds float, float* %".1", i32 %".8"
  %".10" = load float, float* %".9"
  %".11" = mul i32 %".6", 10
  %".12" = add i32 %".11", 1
  %".13" = getelementptr inbounds float, float* %".1", i32 %".12"
  %".14" = load float, float* %".13"
  %".15" = mul i32 %".6", 10
  %".16" = add i32 %".15", 2
  %".17" = getelementptr inbounds float, float* %".1", i32 %".16"
  %".18" = load float, float* %".17"
  %".19" = mul i32 %".6", 10
  %".20" = add i32 %".19", 3
  %".21" = getelementptr inbounds float, float* %".1", i32 %".20"
  %".22" = load float, float* %".21"
  %".23" = mul i32 %".6", 10
  %".24" = add i32 %".23", 4
  %".25" = getelementptr inbounds float, float* %".1", i32 %".24"
  %".26" = load float, float* %".25"
  %".27" = mul i32 %".6", 10
  %".28" = add i32 %".27", 5
  %".29" = getelementptr inbounds float, float* %".1", i32 %".28"
  %".30" = load float, float* %".29"
  %".31" = mul i32 %".6", 10
  %".32" = add i32 %".31", 6
  %".33" = getelementptr inbounds float, float* %".1", i32 %".32"
  %".34" = load float, float* %".33"
  %".35" = mul i32 %".6", 10
  %".36" = add i32 %".35", 7
  %".37" = getelementptr inbounds float, float* %".1", i32 %".36"
  %".38" = load float, float* %".37"
  %".39" = mul i32 %".6", 10
  %".40" = add i32 %".39", 8
  %".41" = getelementptr inbounds float, float* %".1", i32 %".40"
  %".42" = load float, float* %".41"
  %".43" = mul i32 %".6", 10
  %".44" = add i32 %".43", 9
  %".45" = getelementptr inbounds float, float* %".1", i32 %".44"
  %".46" = load float, float* %".45"
  %".47" = fadd float %".10",              0x0
  %".48" = fadd float %".14", %".47"
  %".49" = fadd float %".18", %".48"
  %".50" = fadd float %".22", %".49"
  %".51" = fadd float %".26", %".50"
  %".52" = fadd float %".30", %".51"
  %".53" = fadd float %".34", %".52"
  %".54" = fadd float %".38", %".53"
  %".55" = fadd float %".42", %".54"
  %".56" = fadd float %".46", %".55"
  %".57" = getelementptr inbounds float, float* %".2", i32 %".6"
  store float %".56", float* %".57"
  %".59" = add i32 %".6", 1
  %".60" = icmp eq i32 %".59", 10
  br i1 %".60", label %"out", label %"localidx"
out:
  ret void
}

@"buf_7" = dso_local global [10 x float] zeroinitializer, align 16
define void @"UnaryOPS.LOG_10_1"(float* %".1", float* %".2")
{
entry:
  br label %"loop"
loop:
  %".6" = phi  i32 [0, %"entry"], [%".12", %"loop"]
  %".7" = getelementptr float, float* %".1", i32 %".6"
  %".8" = load float, float* %".7"
  %".9" = getelementptr float, float* %".2", i32 %".6"
  %".10" = call float @"llvm.log.f32"(float %".8")
  store float %".10", float* %".9"
  %".12" = add i32 %".6", 1
  %".13" = icmp eq i32 %".12", 10
  br i1 %".13", label %"out", label %"loop"
out:
  ret void
}

declare float @"llvm.log.f32"(float %".1")

@"buf_8" = dso_local global [100 x float] zeroinitializer, align 16
@"buf_9" = dso_local global [100 x float] zeroinitializer, align 16