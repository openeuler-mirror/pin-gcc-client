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

   Author: Guangya Ding
   Create: 2022-11-23
   Description:
    This file defines the types for the Plugin dialect in MLIR. These MLIR types
    correspond to the Plugin IR type system.
*/

#ifndef MLIR_DIALECT_PLUGINIR_PLUGINTYPES_H_
#define MLIR_DIALECT_PLUGINIR_PLUGINTYPES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace PluginIR {
using namespace mlir;

enum PluginTypeID {
    UndefTyID = 0,
    // PrimitiveTypes
    VoidTyID,      ///< type with no size
    UIntegerTy1ID,  ///< 1-bit unsigned integer type
    UIntegerTy8ID,  ///< 8-bit unsigned integer type
    UIntegerTy16ID,  ///< 16-bit unsigned integer type
    UIntegerTy32ID,  ///< 32-bit unsigned integer type
    UIntegerTy64ID,  ///< 64-bit unsigned integer type
    IntegerTy1ID,  ///< 1-bit signed integer type
    IntegerTy8ID,  ///< 8-bit signed integer type
    IntegerTy16ID,  ///< 16-bit signed integer type
    IntegerTy32ID,  ///< 32-bit signed integer type
    IntegerTy64ID,  ///< 64-bit signed integer type
    BooleanTyID,    ///< 1-bit signless integer type
    FloatTyID,     ///< 32-bit floating point type
    DoubleTyID,    ///< 64-bit floating point type

    // Derived types
    FunctionTyID,      ///< Functions
    PointerTyID,       ///< Pointers
    StructTyID,        ///< Structures
    ArrayTyID,         ///< Arrays
};

class PluginTypeBase : public Type {
public:
    using Type::Type;

    PluginTypeID getPluginTypeID ();
    unsigned getPluginIntOrFloatBitWidth ();
    bool isSignedPluginInteger ();
    bool isUnsignedPluginInteger ();
    void setTypeSize (unsigned size);
    void setReadOnlyFlag (unsigned readOnlyFlag);
    unsigned getTypeSize ();
    unsigned getReadOnlyFlag ();

private:
    unsigned size;
    unsigned readOnlyFlag;
}; // class PluginTypeBase

namespace detail {
    struct PluginIntegerTypeStorage;
    struct PluginFloatTypeStorage;
}

class PluginIntegerType : public Type::TypeBase<PluginIntegerType, PluginTypeBase, detail::PluginIntegerTypeStorage> {
public:
    using Base::Base;

    enum SignednessSemantics {
        Signless, /// No signedness semantics
        Signed,   /// Signed integer
        Unsigned, /// Unsigned integer
    };
    /// Return true if this is a signless integer type.
    bool isSignless() const { return getSignedness() == Signless; }
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getSignedness() == Signed; }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return getSignedness() == Unsigned; }

    PluginTypeID getPluginTypeID ();

    static PluginIntegerType get(MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);

    unsigned getWidth() const;
    SignednessSemantics getSignedness() const;
};

class PluginFloatType : public Type::TypeBase<PluginFloatType, PluginTypeBase, detail::PluginFloatTypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static PluginFloatType get(MLIRContext *context, unsigned width);

    unsigned getWidth() const;
};

class PluginVoidType : public Type::TypeBase<PluginVoidType, PluginTypeBase, TypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

}; // class PluginVoidType

class PluginUndefType : public Type::TypeBase<PluginUndefType, PluginTypeBase, TypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

}; // class PluginUndefType

class PluginBooleanType : public Type::TypeBase<PluginBooleanType, PluginTypeBase, TypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

}; // class PluginBooleanType

//===----------------------------------------------------------------------===//
// Printing and parsing. TODO
//===----------------------------------------------------------------------===//

} // namespace PluginIR

#endif // MLIR_DIALECT_PLUGINIR_PLUGINTYPES_H_