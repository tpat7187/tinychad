; ModuleID = ""
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"main"(float* %".1", float* %".2", float* %".3", float* %".4", float* %".5", float* %".6", float* %".7", float* %".8", float* %".9", float* %".10")
{
main:
  call void @"MAX_(20, 20)_(20, 1)"(float* %".2", float* %".3")
  call void @"CAST_(20, 1)_(20, 20)"(float* %".3", float* %".4")
  call void @"SUB_(20, 20)_(20, 20)"(float* %".2", float* %".4", float* %".5")
  call void @"EXP_(20, 20)_(20, 20)"(float* %".5", float* %".6")
  call void @"SUM_(20, 20)_(20, 1)"(float* %".6", float* %".7")
  call void @"LOG_(20, 1)_(20, 1)"(float* %".7", float* %".8")
  call void @"CAST_(20, 1)_(20, 20)"(float* %".8", float* %".9")
  call void @"SUB_(20, 20)_(20, 20)"(float* %".5", float* %".9", float* %".10")
  br label %"exit"
exit:
  ret void
}

define void @"MAX_(20, 20)_(20, 1)"(float* %".1", float* %".2")
{
entry:
  br label %"localidx"
localidx:
  %".6" = phi  i32 [0, %"entry"], [%".128", %"localidx"]
  %".7" = mul i32 %".6", 20
  %".8" = add i32 %".7", 0
  %".9" = getelementptr inbounds float, float* %".1", i32 %".8"
  %".10" = load float, float* %".9"
  %".11" = mul i32 %".6", 20
  %".12" = add i32 %".11", 1
  %".13" = getelementptr inbounds float, float* %".1", i32 %".12"
  %".14" = load float, float* %".13"
  %".15" = mul i32 %".6", 20
  %".16" = add i32 %".15", 2
  %".17" = getelementptr inbounds float, float* %".1", i32 %".16"
  %".18" = load float, float* %".17"
  %".19" = mul i32 %".6", 20
  %".20" = add i32 %".19", 3
  %".21" = getelementptr inbounds float, float* %".1", i32 %".20"
  %".22" = load float, float* %".21"
  %".23" = mul i32 %".6", 20
  %".24" = add i32 %".23", 4
  %".25" = getelementptr inbounds float, float* %".1", i32 %".24"
  %".26" = load float, float* %".25"
  %".27" = mul i32 %".6", 20
  %".28" = add i32 %".27", 5
  %".29" = getelementptr inbounds float, float* %".1", i32 %".28"
  %".30" = load float, float* %".29"
  %".31" = mul i32 %".6", 20
  %".32" = add i32 %".31", 6
  %".33" = getelementptr inbounds float, float* %".1", i32 %".32"
  %".34" = load float, float* %".33"
  %".35" = mul i32 %".6", 20
  %".36" = add i32 %".35", 7
  %".37" = getelementptr inbounds float, float* %".1", i32 %".36"
  %".38" = load float, float* %".37"
  %".39" = mul i32 %".6", 20
  %".40" = add i32 %".39", 8
  %".41" = getelementptr inbounds float, float* %".1", i32 %".40"
  %".42" = load float, float* %".41"
  %".43" = mul i32 %".6", 20
  %".44" = add i32 %".43", 9
  %".45" = getelementptr inbounds float, float* %".1", i32 %".44"
  %".46" = load float, float* %".45"
  %".47" = mul i32 %".6", 20
  %".48" = add i32 %".47", 10
  %".49" = getelementptr inbounds float, float* %".1", i32 %".48"
  %".50" = load float, float* %".49"
  %".51" = mul i32 %".6", 20
  %".52" = add i32 %".51", 11
  %".53" = getelementptr inbounds float, float* %".1", i32 %".52"
  %".54" = load float, float* %".53"
  %".55" = mul i32 %".6", 20
  %".56" = add i32 %".55", 12
  %".57" = getelementptr inbounds float, float* %".1", i32 %".56"
  %".58" = load float, float* %".57"
  %".59" = mul i32 %".6", 20
  %".60" = add i32 %".59", 13
  %".61" = getelementptr inbounds float, float* %".1", i32 %".60"
  %".62" = load float, float* %".61"
  %".63" = mul i32 %".6", 20
  %".64" = add i32 %".63", 14
  %".65" = getelementptr inbounds float, float* %".1", i32 %".64"
  %".66" = load float, float* %".65"
  %".67" = mul i32 %".6", 20
  %".68" = add i32 %".67", 15
  %".69" = getelementptr inbounds float, float* %".1", i32 %".68"
  %".70" = load float, float* %".69"
  %".71" = mul i32 %".6", 20
  %".72" = add i32 %".71", 16
  %".73" = getelementptr inbounds float, float* %".1", i32 %".72"
  %".74" = load float, float* %".73"
  %".75" = mul i32 %".6", 20
  %".76" = add i32 %".75", 17
  %".77" = getelementptr inbounds float, float* %".1", i32 %".76"
  %".78" = load float, float* %".77"
  %".79" = mul i32 %".6", 20
  %".80" = add i32 %".79", 18
  %".81" = getelementptr inbounds float, float* %".1", i32 %".80"
  %".82" = load float, float* %".81"
  %".83" = mul i32 %".6", 20
  %".84" = add i32 %".83", 19
  %".85" = getelementptr inbounds float, float* %".1", i32 %".84"
  %".86" = load float, float* %".85"
  %".87" = fadd float %".10",              0x0
  %".88" = fcmp ugt float %".14", %".87"
  %".89" = select  i1 %".88", float %".14", float %".87"
  %".90" = fcmp ugt float %".18", %".89"
  %".91" = select  i1 %".90", float %".18", float %".89"
  %".92" = fcmp ugt float %".22", %".91"
  %".93" = select  i1 %".92", float %".22", float %".91"
  %".94" = fcmp ugt float %".26", %".93"
  %".95" = select  i1 %".94", float %".26", float %".93"
  %".96" = fcmp ugt float %".30", %".95"
  %".97" = select  i1 %".96", float %".30", float %".95"
  %".98" = fcmp ugt float %".34", %".97"
  %".99" = select  i1 %".98", float %".34", float %".97"
  %".100" = fcmp ugt float %".38", %".99"
  %".101" = select  i1 %".100", float %".38", float %".99"
  %".102" = fcmp ugt float %".42", %".101"
  %".103" = select  i1 %".102", float %".42", float %".101"
  %".104" = fcmp ugt float %".46", %".103"
  %".105" = select  i1 %".104", float %".46", float %".103"
  %".106" = fcmp ugt float %".50", %".105"
  %".107" = select  i1 %".106", float %".50", float %".105"
  %".108" = fcmp ugt float %".54", %".107"
  %".109" = select  i1 %".108", float %".54", float %".107"
  %".110" = fcmp ugt float %".58", %".109"
  %".111" = select  i1 %".110", float %".58", float %".109"
  %".112" = fcmp ugt float %".62", %".111"
  %".113" = select  i1 %".112", float %".62", float %".111"
  %".114" = fcmp ugt float %".66", %".113"
  %".115" = select  i1 %".114", float %".66", float %".113"
  %".116" = fcmp ugt float %".70", %".115"
  %".117" = select  i1 %".116", float %".70", float %".115"
  %".118" = fcmp ugt float %".74", %".117"
  %".119" = select  i1 %".118", float %".74", float %".117"
  %".120" = fcmp ugt float %".78", %".119"
  %".121" = select  i1 %".120", float %".78", float %".119"
  %".122" = fcmp ugt float %".82", %".121"
  %".123" = select  i1 %".122", float %".82", float %".121"
  %".124" = fcmp ugt float %".86", %".123"
  %".125" = select  i1 %".124", float %".86", float %".123"
  %".126" = getelementptr inbounds float, float* %".2", i32 %".6"
  store float %".125", float* %".126"
  %".128" = add i32 %".6", 1
  %".129" = icmp eq i32 %".128", 20
  br i1 %".129", label %"out", label %"localidx"
out:
  ret void
}

define void @"CAST_(20, 1)_(20, 20)"(float* %".1", float* %".2")
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
  %".10" = mul i32 %".4", 20
  %".11" = add i32 %".10", %".5"
  %".12" = getelementptr float, float* %".2", i32 %".11"
  store float %".9", float* %".12"
  %".14" = add i32 %".5", 1
  %".15" = icmp eq i32 %".14", 20
  br i1 %".15", label %"globalidx_edit", label %"localidx"
globalidx_edit:
  %".17" = add i32 %".4", 1
  %".18" = icmp eq i32 %".17", 20
  br i1 %".18", label %"out", label %"globalidx"
out:
  ret void
}

define void @"SUB_(20, 20)_(20, 20)"(float* %".1", float* %".2", float* %".3")
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
  %".16" = icmp ult i32 %".7", 400
  br i1 %".16", label %"loop", label %"out"
out:
  ret void
}

define void @"EXP_(20, 20)_(20, 20)"(float* %".1", float* %".2")
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
  %".13" = icmp ult i32 %".6", 400
  br i1 %".13", label %"loop", label %"out"
out:
  ret void
}

declare float @"llvm.exp.f32"(float %".1")

define void @"SUM_(20, 20)_(20, 1)"(float* %".1", float* %".2")
{
entry:
  br label %"localidx"
localidx:
  %".6" = phi  i32 [0, %"entry"], [%".109", %"localidx"]
  %".7" = mul i32 %".6", 20
  %".8" = add i32 %".7", 0
  %".9" = getelementptr inbounds float, float* %".1", i32 %".8"
  %".10" = load float, float* %".9"
  %".11" = mul i32 %".6", 20
  %".12" = add i32 %".11", 1
  %".13" = getelementptr inbounds float, float* %".1", i32 %".12"
  %".14" = load float, float* %".13"
  %".15" = mul i32 %".6", 20
  %".16" = add i32 %".15", 2
  %".17" = getelementptr inbounds float, float* %".1", i32 %".16"
  %".18" = load float, float* %".17"
  %".19" = mul i32 %".6", 20
  %".20" = add i32 %".19", 3
  %".21" = getelementptr inbounds float, float* %".1", i32 %".20"
  %".22" = load float, float* %".21"
  %".23" = mul i32 %".6", 20
  %".24" = add i32 %".23", 4
  %".25" = getelementptr inbounds float, float* %".1", i32 %".24"
  %".26" = load float, float* %".25"
  %".27" = mul i32 %".6", 20
  %".28" = add i32 %".27", 5
  %".29" = getelementptr inbounds float, float* %".1", i32 %".28"
  %".30" = load float, float* %".29"
  %".31" = mul i32 %".6", 20
  %".32" = add i32 %".31", 6
  %".33" = getelementptr inbounds float, float* %".1", i32 %".32"
  %".34" = load float, float* %".33"
  %".35" = mul i32 %".6", 20
  %".36" = add i32 %".35", 7
  %".37" = getelementptr inbounds float, float* %".1", i32 %".36"
  %".38" = load float, float* %".37"
  %".39" = mul i32 %".6", 20
  %".40" = add i32 %".39", 8
  %".41" = getelementptr inbounds float, float* %".1", i32 %".40"
  %".42" = load float, float* %".41"
  %".43" = mul i32 %".6", 20
  %".44" = add i32 %".43", 9
  %".45" = getelementptr inbounds float, float* %".1", i32 %".44"
  %".46" = load float, float* %".45"
  %".47" = mul i32 %".6", 20
  %".48" = add i32 %".47", 10
  %".49" = getelementptr inbounds float, float* %".1", i32 %".48"
  %".50" = load float, float* %".49"
  %".51" = mul i32 %".6", 20
  %".52" = add i32 %".51", 11
  %".53" = getelementptr inbounds float, float* %".1", i32 %".52"
  %".54" = load float, float* %".53"
  %".55" = mul i32 %".6", 20
  %".56" = add i32 %".55", 12
  %".57" = getelementptr inbounds float, float* %".1", i32 %".56"
  %".58" = load float, float* %".57"
  %".59" = mul i32 %".6", 20
  %".60" = add i32 %".59", 13
  %".61" = getelementptr inbounds float, float* %".1", i32 %".60"
  %".62" = load float, float* %".61"
  %".63" = mul i32 %".6", 20
  %".64" = add i32 %".63", 14
  %".65" = getelementptr inbounds float, float* %".1", i32 %".64"
  %".66" = load float, float* %".65"
  %".67" = mul i32 %".6", 20
  %".68" = add i32 %".67", 15
  %".69" = getelementptr inbounds float, float* %".1", i32 %".68"
  %".70" = load float, float* %".69"
  %".71" = mul i32 %".6", 20
  %".72" = add i32 %".71", 16
  %".73" = getelementptr inbounds float, float* %".1", i32 %".72"
  %".74" = load float, float* %".73"
  %".75" = mul i32 %".6", 20
  %".76" = add i32 %".75", 17
  %".77" = getelementptr inbounds float, float* %".1", i32 %".76"
  %".78" = load float, float* %".77"
  %".79" = mul i32 %".6", 20
  %".80" = add i32 %".79", 18
  %".81" = getelementptr inbounds float, float* %".1", i32 %".80"
  %".82" = load float, float* %".81"
  %".83" = mul i32 %".6", 20
  %".84" = add i32 %".83", 19
  %".85" = getelementptr inbounds float, float* %".1", i32 %".84"
  %".86" = load float, float* %".85"
  %".87" = fadd float %".10",              0x0
  %".88" = fadd float %".14", %".87"
  %".89" = fadd float %".18", %".88"
  %".90" = fadd float %".22", %".89"
  %".91" = fadd float %".26", %".90"
  %".92" = fadd float %".30", %".91"
  %".93" = fadd float %".34", %".92"
  %".94" = fadd float %".38", %".93"
  %".95" = fadd float %".42", %".94"
  %".96" = fadd float %".46", %".95"
  %".97" = fadd float %".50", %".96"
  %".98" = fadd float %".54", %".97"
  %".99" = fadd float %".58", %".98"
  %".100" = fadd float %".62", %".99"
  %".101" = fadd float %".66", %".100"
  %".102" = fadd float %".70", %".101"
  %".103" = fadd float %".74", %".102"
  %".104" = fadd float %".78", %".103"
  %".105" = fadd float %".82", %".104"
  %".106" = fadd float %".86", %".105"
  %".107" = getelementptr inbounds float, float* %".2", i32 %".6"
  store float %".106", float* %".107"
  %".109" = add i32 %".6", 1
  %".110" = icmp eq i32 %".109", 20
  br i1 %".110", label %"out", label %"localidx"
out:
  ret void
}

define void @"LOG_(20, 1)_(20, 1)"(float* %".1", float* %".2")
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
  %".13" = icmp ult i32 %".6", 20
  br i1 %".13", label %"loop", label %"out"
out:
  ret void
}

declare float @"llvm.log.f32"(float %".1")
