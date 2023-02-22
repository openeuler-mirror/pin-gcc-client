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
// ===----------------------------------------------------------------------===//
//
// This file defines operations in the Plugin dialect.
//
// ===----------------------------------------------------------------------===//

#include "Dialect/PluginDialect.h"
#include "Dialect/PluginOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::Plugin;

void FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       uint64_t id, StringRef funcName, bool declaredInline)
{
    FunctionOp::build(builder, state,
        builder.getI64IntegerAttr(id),
        builder.getStringAttr(funcName),
        builder.getBoolAttr(declaredInline));
}

void LocalDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    uint64_t id, StringRef symName, int64_t typeID, uint64_t typeWidth)
{
    LocalDeclOp::build(builder, state,
        builder.getI64IntegerAttr(id),
        builder.getStringAttr(symName),
        builder.getI64IntegerAttr(typeID),
        builder.getI64IntegerAttr(typeWidth));
}

void LoopOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    uint64_t id, uint32_t index, uint64_t innerLoopId, uint64_t outerLoopId, uint32_t numBlock)
{
    LoopOp::build(builder, state,
        builder.getI64IntegerAttr(id),
        builder.getI32IntegerAttr(index),
        builder.getI64IntegerAttr(innerLoopId),
        builder.getI64IntegerAttr(outerLoopId),
        builder.getI32IntegerAttr(numBlock));
}

// ===----------------------------------------------------------------------===//
// PlaceholderOp

void PlaceholderOp::build(OpBuilder &builder, OperationState &state,
    uint64_t id, IDefineCode defCode, bool readOnly, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addTypes(retType);
}

// ===----------------------------------------------------------------------===//
// MemOp

void MemOp::build(OpBuilder &builder, OperationState &state,
    uint64_t id, IDefineCode defCode, bool readOnly, Value addr, Value offset, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addOperands({addr, offset});
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    if (retType) state.addTypes(retType);
}

// ===----------------------------------------------------------------------===//
// SSAOp

void SSAOp::build(OpBuilder &builder, OperationState &state, uint64_t id, IDefineCode defCode, bool readOnly,
    uint64_t nameVarId, uint64_t ssaParmDecl, uint64_t version, uint64_t definingId, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("nameVarId", builder.getI64IntegerAttr(nameVarId));
    state.addAttribute("ssaParmDecl", builder.getI64IntegerAttr(ssaParmDecl));
    state.addAttribute("version", builder.getI64IntegerAttr(version));
    state.addAttribute("definingId", builder.getI64IntegerAttr(definingId));
    state.addTypes(retType);
}

// ===----------------------------------------------------------------------===//
// ConstOp

void ConstOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
    IDefineCode defCode, bool readOnly, Attribute init, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("init", init);
    if (retType) state.addTypes(retType);
}

// ===----------------------------------------------------------------------===//
// PointerOp

void PointerOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
    IDefineCode defCode, bool readOnly, Type retType, bool pointeeReadOnly)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addTypes(retType);
    state.addAttribute("pointeeReadOnly", builder.getBoolAttr(pointeeReadOnly));
}

//===----------------------------------------------------------------------===//
// DeclBaseOp

void DeclBaseOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, bool addressable, bool used, int32_t uid, Value initial,
                      Value name, Optional<uint64_t> chain, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("addressable", builder.getBoolAttr(addressable));
    state.addAttribute("used", builder.getBoolAttr(used));
    state.addAttribute("uid", builder.getI32IntegerAttr(uid));
    state.addOperands(initial);
    if(chain) {
        state.addAttribute("chain", builder.getI64IntegerAttr(chain.getValue()));
    }
    state.addOperands(name);
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// BlockOp

void BlockOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, Optional<Value> vars, Optional<uint64_t> supercontext,
                      Optional<Value> subblocks, Optional<Value> abstract_origin, Optional<Value> chain, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    if(vars) {
        state.addOperands(vars.getValue());
    }
    if(supercontext) {
        state.addAttribute("supercontext", builder.getI64IntegerAttr(supercontext.getValue()));
    }
    if(subblocks) {
        state.addOperands(subblocks.getValue());
    }
    if(abstract_origin) {
        state.addOperands(abstract_origin.getValue());
    }
    if(chain) {
        state.addOperands(chain.getValue());
    }
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// VecOp

void VecOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, int32_t len, ArrayRef<Value> elements, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("len", builder.getI32IntegerAttr(len));
    state.addOperands(elements);
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// ConstructorOp

void ConstructorOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, int32_t len, ArrayRef<Value> idx,
                      ArrayRef<Value> val, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("len", builder.getI32IntegerAttr(len));
    state.addOperands(idx);
    state.addOperands(val);

    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// FieldDeclOp

void FieldDeclOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, bool addressable, bool used, int32_t uid, Value initial,
                      Value name, uint64_t chain, Value fieldOffset, Value fieldBitOffset, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("addressable", builder.getBoolAttr(addressable));
    state.addAttribute("used", builder.getBoolAttr(used));
    state.addAttribute("uid", builder.getI32IntegerAttr(uid));
    state.addOperands(initial);
    state.addAttribute("chain", builder.getI64IntegerAttr(chain));
    state.addOperands(name);
    state.addOperands({fieldOffset, fieldBitOffset});
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// AddressOp

void AddressOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, Value operand, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addOperands(operand);
    state.addTypes(retType);
}
//===----------------------------------------------------------------------===//
// ListOp

void ListOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                  IDefineCode defCode, bool readOnly, bool hasPurpose,
                ArrayRef<Value> operands, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("hasPurpose", builder.getBoolAttr(hasPurpose));
    state.addOperands(operands);
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// StrOp

void StrOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                  IDefineCode defCode, bool readOnly, StringRef str, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("str", builder.getStringAttr(str));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// ArrayOp

void ArrayOp::build(OpBuilder &builder, OperationState &state,
                  uint64_t id, IDefineCode defCode, bool readOnly,
                  Value addr, Value offset, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addOperands({addr, offset});
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    if (retType) state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// ComponentOp

void ComponentOp::build(OpBuilder &builder, OperationState &state,
                  uint64_t id, IDefineCode defCode, bool readOnly,
                  Value component, Value field, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));

    state.addOperands({component, field});
    if (retType) state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// CallOp

void CallOp::build(OpBuilder &builder, OperationState &state,
    uint64_t id, StringRef callee, ArrayRef<Value> arguments, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    state.addAttribute("callee", builder.getSymbolRefAttr(callee));
    if (retType != nullptr) state.addTypes(retType);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   uint64_t id, ArrayRef<Value> arguments, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    if (retType != nullptr) state.addTypes(retType);
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable CallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range CallOp::getArgOperands()
{
    return inputs();
}

// ===----------------------------------------------------------------------===//
// CondOp

void CondOp::build(OpBuilder &builder, OperationState &state, uint64_t id, uint64_t address,
    IComparisonCode condCode, Value lhs, Value rhs, Block* tb, Block* fb, uint64_t tbaddr,
    uint64_t fbaddr, Value trueLabel, Value falseLabel)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addOperands({lhs, rhs});
    state.addAttribute("tbaddr", builder.getI64IntegerAttr(tbaddr));
    state.addAttribute("fbaddr", builder.getI64IntegerAttr(fbaddr));
    state.addSuccessors(tb);
    state.addSuccessors(fb);
    state.addAttribute("condCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(condCode)));
    if (trueLabel != nullptr) state.addOperands(trueLabel);
    if (falseLabel != nullptr) state.addOperands(falseLabel);
}

// ===----------------------------------------------------------------------===//
// PhiOp

void PhiOp::build(OpBuilder &builder, OperationState &state, uint64_t id, uint32_t capacity,
    uint32_t nArgs, ArrayRef<Value> operands, Type resultType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("capacity", builder.getI32IntegerAttr(capacity));
    state.addAttribute("nArgs", builder.getI32IntegerAttr(nArgs));
    state.addOperands(operands);
    if (resultType != nullptr) {
        state.addTypes(resultType);
    }
}

// ===----------------------------------------------------------------------===//
// AssignOp

void AssignOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
    IExprCode exprCode, ArrayRef<Value> operands, Type resultType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("exprCode", builder.getI32IntegerAttr(static_cast<int32_t>(exprCode)));
    state.addOperands(operands);
    if (resultType != nullptr) state.addTypes(resultType);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseAssignOp(mlir::OpAsmParser &parser, mlir::OperationState &result)
{
    mlir::DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes)) {
        return failure();
    }

    result.addTypes(value.getType());
    return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, AssignOp op)
{
    printer << "assign ";
    printer.printType(op.getType());
}

// ===----------------------------------------------------------------------===//
// BaseOp

void BaseOp::build(OpBuilder &builder, OperationState &state, uint64_t id, StringRef opCode)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("opCode", builder.getStringAttr(opCode));
}

//===----------------------------------------------------------------------===//
// DebugOp

void DebugOp::build(OpBuilder &builder, OperationState &state,
                    uint64_t id)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
}

// ===----------------------------------------------------------------------===//
// FallThroughOp

void FallThroughOp::build(OpBuilder &builder, OperationState &state, uint64_t address, Block* dest, uint64_t destaddr)
{
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("destaddr", builder.getI64IntegerAttr(destaddr));
    state.addSuccessors(dest);
}

// ===----------------------------------------------------------------------===//
// RetOp

void RetOp::build(OpBuilder &builder, OperationState &state, uint64_t address)
{
    state.addAttribute("address", builder.getI64IntegerAttr(address));
}

// ===----------------------------------------------------------------------===//
// TableGen'd op method definitions
// ===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/PluginOps.cpp.inc"