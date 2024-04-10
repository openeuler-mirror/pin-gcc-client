/* Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 3, or (at your option) any
   later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; see the file COPYING3.  If not see
   <http://www.gnu.org/licenses/>.


*/

#ifndef PLUGIN_OPS_TD
#define PLUGIN_OPS_TD

include "PluginDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def CGnodeOp : Plugin_Op<"callgraphnode", [NoMemoryEffect]> {
  let summary = "callgraph node operation";
  let description = [{
    TODO.
  }];

  let arguments = (ins UI64Attr:$id,
                       StrAttr:$symbolName,
                       OptionalAttr<BoolAttr>:$definition,
                       UI32Attr:$order);
  let regions = (region AnyRegion:$bodyRegion);

  // Add custom build methods for the operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<Op>(...)`.
  let builders = [
    OpBuilder<(ins "uint64_t":$id,
                      "StringRef":$symbolName,
                      "bool":$definition,
                      "uint32_t":$order)>
  ];
}

def FunctionOp : Plugin_Op<"function", [NoMemoryEffect]> {
  let summary = "function with a region";
  let description = [{
    TODO.
  }];

  let arguments = (ins UI64Attr:$id,
                       StrAttr:$funcName,
                       OptionalAttr<BoolAttr>:$declaredInline,
                       TypeAttr:$type,
                       OptionalAttr<BoolAttr>:$validType);
  let regions = (region AnyRegion:$bodyRegion);

  // Add custom build methods for the operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<Op>(...)`.
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "StringRef":$funcName, "bool":$declaredInline, "Type":$type, "bool":$validType)>,
    OpBuilder<(ins "uint64_t":$id, "StringRef":$funcName, "bool":$declaredInline, "bool":$validType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType();
  }];
}

def LocalDeclOp : Plugin_Op<"declaration", [NoMemoryEffect]> {
  let summary = "local variable declaration";
  let description = [{TODO}];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<StrAttr>:$symName,
                       OptionalAttr<I64Attr>:$typeID,
                       OptionalAttr<UI64Attr>:$typeWidth);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "StringRef":$symName,
                  "int64_t":$typeID, "uint64_t":$typeWidth)>
  ];
}

def LoopOp : Plugin_Op<"loop", [NoMemoryEffect]> {
  let summary = "loop operation";
  let description = [{
        TODO.
  }];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<UI32Attr>:$index,
                       OptionalAttr<UI64Attr>:$innerLoopId,
                       OptionalAttr<UI64Attr>:$outerLoopId,
                       OptionalAttr<UI32Attr>:$numBlock);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint32_t":$index,
                      "uint64_t":$innerLoopId, "uint64_t":$outerLoopId,
                      "uint32_t":$numBlock)>
  ];
}

def CallOp : Plugin_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "call operation";
  let description = [{
    CallOp represent calls to a user defined function that needs to
    be specialized for the shape of its arguments.
    The callee name is attached as a symbol reference via an attribute.
    The arguments list must match the arguments expected by the callee.
  }];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address,
                       OptionalAttr<FlatSymbolRefAttr>:$callee,
                       Variadic<AnyType>:$inputs);
  let results = (outs Optional<AnyType>:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address, "StringRef":$callee,
                      "ArrayRef<Value>":$arguments, "Type":$retType)>,
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address,
                      "ArrayRef<Value>":$arguments,
                      "Type":$retType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
  }];
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def PhiOp : Plugin_Op<"phi", [NoMemoryEffect]> {
  let summary = "phi op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       UI32Attr:$capacity,
                       UI32Attr:$nArgs,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint32_t":$capacity, "uint32_t":$nArgs,
                  "ArrayRef<Value>":$operands, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetArgDef(int i) { return getOperand(i); }
  }];
}

def AssignOp : Plugin_Op<"assign", [NoMemoryEffect]> {
  let summary = "assign op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       IExprCodeAttr:$exprCode,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IExprCode":$exprCode,
                  "ArrayRef<Value>":$operands, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
  }];
  // Specify a parser and printer method.
  // let parser = [{ return ::parseAssignOp(parser, result); }];
  // let printer = [{ return ::print(p, *this); }];
}

def NopOp : Plugin_Op<"nop", [NoMemoryEffect]> {
  let summary = "nop";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id)>,
  ];
}

def EHElseOp : Plugin_Op<"eh_else", [NoMemoryEffect]> {
  let summary = "EHElseOP";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, ArrayAttr:$nBody, ArrayAttr:$eBody);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "ArrayRef<uint64_t>":$nBody, "ArrayRef<uint64_t>":$eBody)>,
  ];
}

class Placeholder<string mnemonic> : Plugin_Op<mnemonic, [SameVariadicOperandSize]> {
  dag commonArgs = (ins UI64Attr:$id,
                        OptionalAttr<IDefineCodeAttr>:$defCode,
                        OptionalAttr<BoolAttr>:$readOnly);
  let results = (outs AnyType);
  let arguments = commonArgs;
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
  }];
}

class Pointer<string mnemonic> : Placeholder<mnemonic> {
  dag PointerArgs = (ins BoolAttr:$PointeeReadOnly);
  let arguments = !con(commonArgs, PointerArgs);
}

class SSA<string mnemonic> : Placeholder<mnemonic> {
  dag SSAArgs = (ins UI64Attr:$nameVarId,
                     UI64Attr:$ssaParmDecl,
                     UI64Attr:$version,
                     UI64Attr:$definingId);
  let arguments = !con(commonArgs, SSAArgs);
}

class MemRef<string mnemonic> : Placeholder<mnemonic> {
    dag MemRefArgs = (ins AnyType:$base,
                          AnyType:$offset);
    let arguments = !con(commonArgs, MemRefArgs);
}

class ComponentRef<string mnemonic> : Placeholder<mnemonic> {
    dag ComponentRefArgs = (ins AnyType:$component,
                          AnyType:$field);
    let arguments = !con(commonArgs, ComponentRefArgs);
}

class Constant<string mnemonic> : Placeholder<mnemonic> {
    dag ConstArgs = (ins AnyAttr:$init);
    let arguments = !con(commonArgs, ConstArgs);
}

class List<string mnemonic> : Placeholder<mnemonic> {
    dag ListArgs = (ins BoolAttr:$hasPurpose,
                          Variadic<AnyType>:$operands);
    let arguments = !con(commonArgs, ListArgs);
}

class StringCST<string mnemonic> : Placeholder<mnemonic> {
    dag StringArgs = (ins StrAttr:$str);
    let arguments = !con(commonArgs, StringArgs);
}

class ArrayRef<string mnemonic> : Placeholder<mnemonic> {
    dag ArrayRefArgs = (ins AnyType:$base,
                          AnyType:$offset);
    let arguments = !con(commonArgs, ArrayRefArgs);
}

class DeclBase<string mnemonic> : Placeholder<mnemonic> {
    dag DeclBaseArgs = (ins BoolAttr:$addressable, BoolAttr:$used, I32Attr:$uid, 
                          AnyType:$initial, AnyType:$name, OptionalAttr<I64Attr>:$chain);
    let arguments = !con(commonArgs, DeclBaseArgs);
}

class FieldDecl<string mnemonic> : DeclBase<mnemonic> {
    dag FieldDeclArgs = (ins AnyType:$fieldOffset, AnyType:$fieldBitOffset);
    let arguments = !con(commonArgs, DeclBaseArgs, FieldDeclArgs);
}

class Address<string mnemonic> : Placeholder<mnemonic> {
  dag AddressArgs = (ins AnyType:$operand);
  let arguments = !con(commonArgs, AddressArgs);
}

class Constructor<string mnemonic> : Placeholder<mnemonic> {
  dag ConstructorArgs = (ins I32Attr:$len, Variadic<AnyType>:$idx, Variadic<AnyType>:$val);
  let arguments = !con(commonArgs, ConstructorArgs);
}

class Vec<string mnemonic> : Placeholder<mnemonic> {
  dag VecArgs = (ins I32Attr:$len, Variadic<AnyType>:$elements);
  let arguments = !con(commonArgs, VecArgs);
}

class BLOCK<string mnemonic> : Placeholder<mnemonic> {
  dag BLOCKArgs = (ins I32Attr:$len, Optional<AnyType>:$vars,  OptionalAttr<I64Attr>:$supercontext,
                    Optional<AnyType>:$subblocks, Optional<AnyType>:$chain, Optional<AnyType>:$abstract_origin);
  let arguments = !con(commonArgs, BLOCKArgs);
}
def PlaceholderOp : Placeholder<"palceholder"> {
  let summary = "PlaceHolder";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "Type":$retType)>
  ];
}

def DeclBaseOp : DeclBase<"decl"> {
  let summary = "DeclBaseOp";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "bool":$addressable, "bool":$used, "int":$uid, "Value":$initial,
                      "Value":$name, "std::optional<uint64_t>":$chain, "Type":$retType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetInitial() { return getInitial(); }
    Value GetName() { return getName(); }
    std::optional<uint64_t> GetChain() { return getChain(); }
  }];
}

def BlockOp : BLOCK<"block"> {
  let summary = "BlockOp";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "std::optional<Value>":$vars, "std::optional<uint64_t>":$supercontext,
                      "std::optional<Value>":$subblocks, "std::optional<Value>":$chain,
                      "std::optional<Value>":$abstract_origin, "Type":$retType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetVars() { return getVars(); }
    std::optional<uint64_t> GetSupercontext() { return getSupercontext(); }
    Value GetSubblocks() { return getSubblocks(); }
    Value GetChain() { return getChain(); }
    Value GetAbstractorigin() { return getAbstractOrigin(); }
  }];
}

def VecOp : Vec<"Vector"> {
  let summary = "VectorOp";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "int":$len, "ArrayRef<Value>":$elements, "Type":$retType)>
  ];
}

def FieldDeclOp : FieldDecl<"field"> {
  let summary = "FieldDeclOp";
  let description = [{TODO}];

  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                    "bool":$readOnly, "bool":$addressable, "bool":$used, "int":$uid, "Value":$initial,
                    "Value":$name, "uint64_t":$chain, "Value":$fieldOffset, "Value":$fieldBitOffset, "Type":$retType)>
];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetInitial() { return getInitial(); }
    Value GetName() { return getName(); }
    std::optional<uint64_t> GetChain() { return getChain(); }
    Value GetFieldOffset() { return getFieldOffset(); }
    Value GetFieldBitOffset() { return getFieldBitOffset(); }
  }];
}

def AddressOp : Address<"address"> {
  let summary = "AddressOp";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                    "bool":$readOnly, "Value":$operand, "Type":$retType)>
];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetOperand() { return getOperand(); }
  }];
}

def ConstructorOp : Constructor<"constructor"> {
  let summary = "ConstructorOp";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                    "bool":$readOnly, "int32_t":$len, "ArrayRef<Value>":$idx, "ArrayRef<Value>":$val,
                    "Type":$retType)>
];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
  }];
}

def ComponentOp : ComponentRef<"component"> {
  let summary = "Component reference op";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "Value":$component, "Value":$field, "Type":$retType)>
  ];
  let results = (outs AnyType);
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetComponent() { return getOperand(0); }
    Value GetField() { return getOperand(1); }
  }];
}

def PointerOp : Pointer<"pointer"> {
  let summary = "pointer";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "Type":$retType,
                      "bool":$pointerReadOnly)>
    ];
}

def MemOp : MemRef<"MemRef"> {
  let summary = "Memory reference op";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "Value":$base, "Value":$offset, "Type":$retType)>
  ];
  let results = (outs AnyType);
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetBase() { return getOperand(0); }
    Value GetOffset() { return getOperand(1); }
  }];
}

def SSAOp : SSA<"SSA"> {
  let summary = "SSA";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "uint64_t":$nameVarId, "uint64_t":$ssaParmDecl, "uint64_t":$version,
                      "uint64_t":$definingId, "Type":$retType)>
  ];
}

def ConstOp : Constant<"constant value"> {
  let summary = "const value";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "Attribute":$init, "Type":$retType)>
  ];
}

def ListOp : List<"ListOp"> {
  let summary = "ListOp";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "bool":$hasPurpose, "ArrayRef<Value>":$operands, "Type":$retType)>
  ];
  let results = (outs AnyType);
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
  }];
}

def StrOp : StringCST<"string"> {
  let summary = "string";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,"StringRef":$str, "Type":$retType)>
  ];
}

def ArrayOp : ArrayRef<"ArrayRef"> {
  let summary = "array ref";
  let description = [{TODO}];
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "Value":$base, "Value":$offset, "Type":$retType)>
  ];
  let results = (outs AnyType);
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetBase() { return getOperand(0); }
    Value GetOffset() { return getOperand(1); }
  }];
}

def BaseOp : Plugin_Op<"statement_base", [NoMemoryEffect]> {
  let summary = "Base operation, just like placeholder for statement.";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, StrAttr:$opCode);
  let results = (outs AnyType);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "StringRef":$opCode)>
  ];
}

def DebugOp : Plugin_Op<"debug", [NoMemoryEffect]> {
  let summary = "DebugOp.";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id);
  let results = (outs AnyType);
  let builders = [
    OpBuilder<(ins "uint64_t":$id)>
  ];
}

def AsmOp : Plugin_Op<"asm", [NoMemoryEffect]> {
  let summary = "asm Op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, StrAttr:$statement, UI32Attr:$nInputs, UI32Attr:$nOutputs, UI32Attr:$nClobbers,
                   Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "StringRef":$statement, "uint32_t":$nInputs, "uint32_t":$nOutputs,
                  "uint32_t":$nClobbers, "ArrayRef<Value>":$operands)>
  ];

  let extraClassDeclaration = [{
    unsigned GetNInputs() { return this->getNInputsAttr().getInt(); };
    unsigned GetNoutputs() { return this->getNOutputsAttr().getInt(); };
    unsigned GetNClobbers() { return this->getNClobbersAttr().getInt(); };
    Value GetInputOp(unsigned index) { return getOperand(index); };
    Value GetOutputOp(unsigned index) { return getOperand(this->getNInputsAttr().getInt() + index); };
    Value GetCkobberOp(unsigned index) { 
      return getOperand(this->getNInputsAttr().getInt() + this->getNOutputsAttr().getInt() + index); };
  }]; 
}

def LabelOp : Plugin_Op<"label", [NoMemoryEffect]> {
  let summary = "Label op.";
  let description = [{TODO}];
  let results = (outs AnyType:$result);
  let arguments = (ins UI64Attr:$id, AnyType:$index);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "Value":$index)>
  ];
  let extraClassDeclaration = [{
    Value GetLabelLabel() { return this->getOperation()->getOperand(0); };
  }];
}

def EHMntOp : Plugin_Op<"eh_mnt", [NoMemoryEffect]> {
  let summary = "EH_MUST_NOT_THROW op.";
  let description = [{TODO}];
  let results = (outs AnyType:$result);
  let arguments = (ins UI64Attr:$id, AnyType:$decl);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "Value":$decl)>
  ];
  let extraClassDeclaration = [{
    Value Getfndecl() { return getDecl(); };
  }];
}

def BindOp : Plugin_Op<"Bind", [NoMemoryEffect]> {
  let summary = "BindOp";
  let description = [{TODO}];
  let results = (outs AnyType:$result);
  let arguments = (ins UI64Attr:$id, AnyType:$vars, ArrayAttr:$body, AnyType:$block);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "Value":$vars, "ArrayRef<uint64_t>":$body, "Value":$block)>
  ];
  let extraClassDeclaration = [{
    Value GetVars() { return getVars(); };
    Value GetBlock() { return getBlock(); };
  }];
}

def TryOp : Plugin_Op<"try", [NoMemoryEffect]> {
  let summary = "TryOp";
  let description = [{TODO}];
  let results = (outs AnyType:$result);
  let arguments = (ins UI64Attr:$id, ArrayAttr:$eval, ArrayAttr:$cleanup, UI64Attr:$kind);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "ArrayRef<uint64_t>":$eval, "ArrayRef<uint64_t>":$cleanup, "uint64_t":$kind)>
  ];
}

def CatchOp : Plugin_Op<"catch", [NoMemoryEffect]> {
  let summary = "CatchOp";
  let description = [{TODO}];
  let results = (outs AnyType:$result);
  let arguments = (ins UI64Attr:$id, AnyType:$types, ArrayAttr:$handler);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "Value":$types, "ArrayRef<uint64_t>":$handler)>
  ];
  let extraClassDeclaration = [{
    Value GetTypes() { return getTypes(); };
  }];
}
// Terminators
// Opaque builder used for terminator operations that contain successors.

class Plugin_TerminatorOp<string mnemonic, list<Trait> traits = []> :
    Plugin_Op<mnemonic, !listconcat(traits, [Terminator])>;

def SwitchOp : Plugin_TerminatorOp<"switch", [NoMemoryEffect]> {
  let summary = "switch op.";
  let description = [{TODO}];
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );
  let arguments = (ins UI64Attr:$id, AnyType:$index, UI64Attr:$address,
                   AnyType:$defaultLabel, Variadic<AnyType>:$args,
                   UI64Attr:$defaultaddr, ArrayAttr:$caseaddrs);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "Value":$index, "uint64_t":$address, "Value":$defaultLabel,
                      "ArrayRef<Value>":$args, "Block*":$defaultDestination, "uint64_t":$defaultaddr,
                      "ArrayRef<Block*>":$caseDestinations, "ArrayRef<uint64_t>":$caseaddrs)>
  ];

  let extraClassDeclaration = [{
    unsigned GetNumLabels() { return getNumOperands() - 2; };
    Block* GetDefautDest() { return getDefaultDestination(); }
    SmallVector<mlir::Block*> GetCaseDest() { return getCaseDestinations(); };
  }];
}

def FallThroughOp : Plugin_TerminatorOp<"fallthrough", [NoMemoryEffect]> {
  let summary = "FallThroughOp";
  let description = [{TODO}];
  let successors = (successor AnySuccessor:$dest);
  // for bb address
  let arguments = (ins UI64Attr:$address, UI64Attr:$destaddr);
  let results = (outs AnyType);
  let builders = [
    OpBuilder<(ins "uint64_t":$address, "Block*":$dest, "uint64_t":$destaddr)>
  ];
}

def CondOp : Plugin_TerminatorOp<"condition", [NoMemoryEffect]> {
  let summary = "condition op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address,
                       IComparisonAttr:$condCode,
                       AnyType:$LHS, AnyType:$RHS,
                       UI64Attr:$tbaddr, UI64Attr:$fbaddr,
                       OptionalAttr<TypeAttr>:$trueLabel,
                       OptionalAttr<TypeAttr>:$falseLabel);
  let successors = (successor AnySuccessor:$tb, AnySuccessor:$fb);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address, "IComparisonCode":$condCode,
                  "Value":$lhs, "Value":$rhs, "Block*":$tb, "Block*":$fb,
                  "uint64_t":$tbaddr, "uint64_t":$fbaddr, "Value":$trueLabel,
                  "Value":$falseLabel)>
  ];
  let extraClassDeclaration = [{
    Value GetLHS() { return getOperand(0); }
    Value GetRHS() { return getOperand(1); }
  }];
}

// todo: currently RetOp do not have a correct assemblyFormat
def RetOp : Plugin_TerminatorOp<"ret", [NoMemoryEffect]> {
  let summary = "RetOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$address); // for bb address
  let results = (outs AnyType);
  let builders = [
    OpBuilder<(ins "uint64_t":$address)>
  ];
}

def GotoOp : Plugin_TerminatorOp<"goto", [NoMemoryEffect]> {
  let summary = "GotoOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address, AnyType:$dest, UI64Attr:$successaddr);
  let results = (outs AnyType);
  let successors = (successor AnySuccessor:$success);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address, "Value":$dest, "Block*":$success, "uint64_t":$successaddr)>
  ];

  let extraClassDeclaration = [{
    Value GetLabel() { return this->getOperation()->getOperand(0); }
  }];
}

def TransactionOp : Plugin_TerminatorOp<"transaction", [NoMemoryEffect]> {
  let summary = "TransactionOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address, ArrayAttr:$stmtaddr, AnyType:$labelNorm,
                      AnyType:$labelUninst, AnyType:$labelOver, UI64Attr:$fallthroughaddr, UI64Attr:$abortaddr);
  let results = (outs AnyType);
  let successors = (successor AnySuccessor:$fallthrough, AnySuccessor:$abort);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address, "ArrayRef<uint64_t>":$stmtaddr, "Value":$labelNorm,
                      "Value":$labelUninst, "Value":$labelOver, "Block*":$fallthrough, "uint64_t":$fallthroughaddr,
                      "Block*":$abort, "uint64_t":$abortaddr)>
  ];

  let extraClassDeclaration = [{
    Value GetTransactionNormal() { return getLabelNorm(); }
    Value GetTransactionUinst() { return getLabelUninst(); }
    Value GetTransactionOver() { return getLabelOver(); }
  }];
}

def ResxOp : Plugin_TerminatorOp<"resx", [NoMemoryEffect]> {
  let summary = "ResxOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address, UI64Attr:$region);
  let results = (outs AnyType);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address, "uint64_t":$region)>
  ];
  let extraClassDeclaration = [{
    uint64_t GetTransactionNormal() { return getRegion(); }
  }];
}

def EHDispatchOp : Plugin_TerminatorOp<"dispatch", [NoMemoryEffect]> {
  let summary = "EHDispatchOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address, UI64Attr:$region, ArrayAttr:$ehHandlersaddrs);
  let results = (outs AnyType);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$ehHandlers);
  let builders = [
    OpBuilder<(ins "uint64_t":$id, "uint64_t":$address, "uint64_t":$region, "ArrayRef<Block*>":$ehHandlers,
                  "ArrayRef<uint64_t>":$ehHandlersaddrs)>
  ];
  let extraClassDeclaration = [{
    uint64_t GetTransactionNormal() { return getRegion(); }
  }];
}
#endif // PLUGIN_OPS_TD
