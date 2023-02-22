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

def FunctionOp : Plugin_Op<"function", [NoSideEffect]> {
  let summary = "function with a region";
  let description = [{
    TODO.
  }];

  let arguments = (ins UI64Attr:$id,
                       StrAttr:$funcName,
                       OptionalAttr<BoolAttr>:$declaredInline);
  let regions = (region AnyRegion:$bodyRegion);

  // Add custom build methods for the operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<Op>(...)`.
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$funcName, "bool":$declaredInline)>
  ];
}

def LocalDeclOp : Plugin_Op<"declaration", [NoSideEffect]> {
  let summary = "local variable declaration";
  let description = [{TODO}];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<StrAttr>:$symName,
                       OptionalAttr<I64Attr>:$typeID,
                       OptionalAttr<UI64Attr>:$typeWidth);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$symName,
                  "int64_t":$typeID, "uint64_t":$typeWidth)>
  ];
}

def LoopOp : Plugin_Op<"loop", [NoSideEffect]> {
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
    OpBuilderDAG<(ins "uint64_t":$id, "uint32_t":$index,
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
  let arguments = (ins UI64Attr:$id,
                       OptionalAttr<FlatSymbolRefAttr>:$callee,
                       Variadic<AnyType>:$inputs);
  let results = (outs Optional<AnyType>:$result);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$callee,
                      "ArrayRef<Value>":$arguments, "Type":$retType)>,
    OpBuilderDAG<(ins "uint64_t":$id,
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

def PhiOp : Plugin_Op<"phi", [NoSideEffect]> {
  let summary = "phi op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       UI32Attr:$capacity,
                       UI32Attr:$nArgs,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "uint32_t":$capacity, "uint32_t":$nArgs,
                  "ArrayRef<Value>":$operands, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetArgDef(int i) { return getOperand(i); }
  }];
}

def AssignOp : Plugin_Op<"assign", [NoSideEffect]> {
  let summary = "assign op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       IExprCodeAttr:$exprCode,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IExprCode":$exprCode,
                  "ArrayRef<Value>":$operands, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
  }];
  // Specify a parser and printer method.
  // let parser = [{ return ::parseAssignOp(parser, result); }];
  // let printer = [{ return ::print(p, *this); }];
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
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "Type":$retType)>
  ];
}

def DeclBaseOp : DeclBase<"decl"> {
  let summary = "DeclBaseOp";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "bool":$addressable, "bool":$used, "int":$uid, "Value":$initial,
                      "Value":$name, "Optional<uint64_t>":$chain, "Type":$retType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetInitial() { return initial(); }
    Value GetName() { return name(); }
    Optional<uint64_t> GetChain() { return chain(); }
  }];
}

def BlockOp : BLOCK<"block"> {
  let summary = "BlockOp";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "Optional<Value>":$vars, "Optional<uint64_t>":$supercontext,
                      "Optional<Value>":$subblocks, "Optional<Value>":$chain,
                      "Optional<Value>":$abstract_origin, "Type":$retType)>
  ];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetVars() { return vars(); }
    Optional<uint64_t> GetSupercontext() { return supercontext(); }
    Value GetSubblocks() { return subblocks(); }
    Value GetChain() { return chain(); }
    Value GetAbstractorigin() { return abstract_origin(); }
  }];
}

def VecOp : Vec<"Vector"> {
  let summary = "VectorOp";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "int":$len, "ArrayRef<Value>":$elements, "Type":$retType)>
  ];
}

def FieldDeclOp : FieldDecl<"field"> {
  let summary = "FieldDeclOp";
  let description = [{TODO}];

  let builders = [
  OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                    "bool":$readOnly, "bool":$addressable, "bool":$used, "int":$uid, "Value":$initial,
                    "Value":$name, "uint64_t":$chain, "Value":$fieldOffset, "Value":$fieldBitOffset, "Type":$retType)>
];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetInitial() { return initial(); }
    Value GetName() { return name(); }
    Optional<uint64_t> GetChain() { return chain(); }
    Value GetFieldOffset() { return fieldOffset(); }
    Value GetFieldBitOffset() { return fieldBitOffset(); }
  }];
}

def AddressOp : Address<"address"> {
  let summary = "AddressOp";
  let description = [{TODO}];
  let builders = [
  OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                    "bool":$readOnly, "Value":$operand, "Type":$retType)>
];
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetOperand() { return operand(); }
  }];
}

def ConstructorOp : Constructor<"constructor"> {
  let summary = "ConstructorOp";
  let description = [{TODO}];
  let builders = [
  OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
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
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
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
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode,
                      "bool":$readOnly, "Type":$retType,
                      "bool":$pointerReadOnly)>
    ];
}

def MemOp : MemRef<"MemRef"> {
  let summary = "Memory reference op";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
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
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "uint64_t":$nameVarId, "uint64_t":$ssaParmDecl, "uint64_t":$version,
                      "uint64_t":$definingId, "Type":$retType)>
  ];
}

def ConstOp : Constant<"constant value"> {
  let summary = "const value";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "Attribute":$init, "Type":$retType)>
  ];
}

def ListOp : List<"ListOp"> {
  let summary = "ListOp";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
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
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,"StringRef":$str, "Type":$retType)>
  ];
}

def ArrayOp : ArrayRef<"ArrayRef"> {
  let summary = "array ref";
  let description = [{TODO}];
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "bool":$readOnly,
                      "Value":$base, "Value":$offset, "Type":$retType)>
  ];
  let results = (outs AnyType);
  let extraClassDeclaration = [{
    Type getResultType() { return this->getOperation()->getResult(0).getType(); }
    Value GetBase() { return getOperand(0); }
    Value GetOffset() { return getOperand(1); }
  }];
}

def BaseOp : Plugin_Op<"statement_base", [NoSideEffect]> {
  let summary = "Base operation, just like placeholder for statement.";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, StrAttr:$opCode);
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$opCode)>
  ];
}

def DebugOp : Plugin_Op<"debug", [NoSideEffect]> {
  let summary = "DebugOp.";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id);
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id)>
  ];
}

// Terminators
// Opaque builder used for terminator operations that contain successors.

class Plugin_TerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    Plugin_Op<mnemonic, !listconcat(traits, [Terminator])>;

def FallThroughOp : Plugin_TerminatorOp<"fallthrough", [NoSideEffect]> {
  let summary = "FallThroughOp";
  let description = [{TODO}];
  let successors = (successor AnySuccessor:$dest);
  // for bb address
  let arguments = (ins UI64Attr:$address, UI64Attr:$destaddr);
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$address, "Block*":$dest, "uint64_t":$destaddr)>
  ];
}

def CondOp : Plugin_TerminatorOp<"condition", [NoSideEffect]> {
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
    OpBuilderDAG<(ins "uint64_t":$id, "uint64_t":$address, "IComparisonCode":$condCode,
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
def RetOp : Plugin_TerminatorOp<"ret", [NoSideEffect]> {
  let summary = "RetOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$address); // for bb address
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$address)>
  ];
}

#endif // PLUGIN_OPS_TD