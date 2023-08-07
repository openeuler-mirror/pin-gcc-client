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
   Create: 2022-11-29
   Description:
    This file defines the types for the Plugin dialect in MLIR. These MLIR types
    correspond to the Plugin IR type system.
*/

#include "Dialect/PluginDialect.h"
#include "Dialect/PluginTypes.h"

using namespace mlir;
using namespace PluginIR;
using namespace mlir::Plugin;

void PluginDialect::registerTypes() {
    addTypes<PluginIntegerType,
             PluginFloatType,
             PluginPointerType,
             PluginArrayType,
             PluginVectorType,
             PluginFunctionType,
             PluginStructType,
             PluginBooleanType,
             PluginVoidType, 
             PluginUndefType>();
}

namespace PluginIR {
namespace Detail {
    /// Integer Type Storage and Uniquing.
    struct PluginIntegerTypeStorage : public TypeStorage {
        PluginIntegerTypeStorage(unsigned width,
                            PluginIntegerType::SignednessSemantics signedness)
            : width(width), signedness(signedness) {}

        /// The hash key used for uniquing.
        using KeyTy = std::pair<unsigned, PluginIntegerType::SignednessSemantics>;

        static llvm::hash_code hashKey(const KeyTy &key)
        {
            return llvm::hash_value(key);
        }

        bool operator==(const KeyTy &key) const
        {
            return KeyTy(width, signedness) == key;
        }

        static PluginIntegerTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key)
        {
            return new (allocator.allocate<PluginIntegerTypeStorage>())
                PluginIntegerTypeStorage(key.first, key.second);
        }

        unsigned width : 30;
        PluginIntegerType::SignednessSemantics signedness : 2;
    };

    struct PluginFloatTypeStorage : public TypeStorage {
        PluginFloatTypeStorage(unsigned width) : width(width) {}

        /// The hash key used for uniquing.
        using KeyTy = unsigned;

        bool operator==(const KeyTy &key) const
        {
            return KeyTy(width) == key;
        }

        static PluginFloatTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key)
        {
            return new (allocator.allocate<PluginFloatTypeStorage>())
                PluginFloatTypeStorage(key);
        }

        unsigned width : 30;
    };

    struct PluginPointerTypeStorage : public TypeStorage {
        using KeyTy = std::tuple<Type, unsigned>;

        PluginPointerTypeStorage(const KeyTy &key)
            : pointee(std::get<0>(key)), readOnlyPointee(std::get<1>(key)) {}

        static PluginPointerTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key)
        {
            return new (allocator.allocate<PluginPointerTypeStorage>())
                PluginPointerTypeStorage(key);
        }

        bool operator==(const KeyTy &key) const
        {
            return std::make_tuple(pointee, readOnlyPointee) == key;
        }

        Type pointee;
        unsigned readOnlyPointee;
    };

    struct PluginTypeAndSizeStorage : public TypeStorage {
        using KeyTy = std::tuple<Type, unsigned>;

        PluginTypeAndSizeStorage(const KeyTy &key)
            : elementType(std::get<0>(key)), numElements(std::get<1>(key)) {}
        
        static PluginTypeAndSizeStorage *construct(TypeStorageAllocator &allocator, KeyTy key)
        {
            return new (allocator.allocate<PluginTypeAndSizeStorage>())
                PluginTypeAndSizeStorage(key);
        }

        bool operator==(const KeyTy &key) const
        {
            return std::make_tuple(elementType, numElements) == key;
        }

        Type elementType;
        unsigned numElements;
    };

    struct PluginFunctionTypeStorage : public TypeStorage {
        using KeyTy = std::tuple<Type, ArrayRef<Type>>;

        PluginFunctionTypeStorage(Type resultType, ArrayRef<Type> argumentTypes)
            : resultType(resultType), argumentTypes(argumentTypes) {}

        static PluginFunctionTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key)
        {
            return new (allocator.allocate<PluginFunctionTypeStorage>())
                PluginFunctionTypeStorage(std::get<0>(key), allocator.copyInto(std::get<1>(key)));
        }

        static unsigned hashKey(const KeyTy &key) {
            // LLVM doesn't like hashing bools in tuples.
            return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
        }

        bool operator==(const KeyTy &key) const
        {
            return std::make_tuple(resultType, argumentTypes) == key;
        }

        Type resultType;
        ArrayRef<Type> argumentTypes;
    };

    struct PluginStructTypeStorage : public TypeStorage {
        using KeyTy = std::tuple<StringRef, ArrayRef<StringRef>>;

        PluginStructTypeStorage(StringRef name, ArrayRef<StringRef> elemNames)
            : name(name), elemNames(elemNames) {}

        static PluginStructTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key)
        {
            return new (allocator.allocate<PluginStructTypeStorage>())
                PluginStructTypeStorage(std::get<0>(key), allocator.copyInto(std::get<1>(key)));
        }

        static unsigned hashKey(const KeyTy &key) {
            // LLVM doesn't like hashing bools in tuples.
            return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
        }

        bool operator==(const KeyTy &key) const
        {
            return std::make_tuple(name, elemNames) == key;
        }

        StringRef name;
        ArrayRef<StringRef> elemNames;
    };
} // namespace Detail
} // namespace PluginIR


// ===----------------------------------------------------------------------===//
// Plugin TypeBase
// ===----------------------------------------------------------------------===//

PluginTypeID PluginTypeBase::getPluginTypeID ()
{
    if (auto Ty = dyn_cast<PluginIR::PluginIntegerType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginFloatType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginBooleanType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginVoidType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginPointerType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginArrayType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginVectorType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginFunctionType>()) {
        return Ty.getPluginTypeID ();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginStructType>()) {
        return Ty.getPluginTypeID ();
    }
    return PluginTypeID::UndefTyID;
}

unsigned PluginTypeBase::getPluginIntOrFloatBitWidth ()
{
    if (auto Ty = dyn_cast<PluginIR::PluginIntegerType>()) {
        return Ty.getWidth();
    }
    if (auto Ty = dyn_cast<PluginIR::PluginFloatType>()) {
        return Ty.getWidth();
    }
    return 0;
}

bool PluginTypeBase::isSignedPluginInteger ()
{
    if (auto Ty = dyn_cast<PluginIR::PluginIntegerType>()) {
        return Ty.isSigned();
    }
    return false;
}

bool PluginTypeBase::isUnsignedPluginInteger ()
{
    if (auto Ty = dyn_cast<PluginIR::PluginIntegerType>()) {
        return Ty.isUnsigned();
    }
    return false;
}

void PluginTypeBase::setTypeSize (unsigned size)
{
    this->size = size;
}

unsigned PluginTypeBase::getTypeSize ()
{
    return size;
}

// ===----------------------------------------------------------------------===//
// Plugin Integer Type
// ===----------------------------------------------------------------------===//

unsigned PluginIntegerType::getWidth() const
{
    return getImpl()->width;
}

PluginIntegerType::SignednessSemantics PluginIntegerType::getSignedness() const
{
    return getImpl()->signedness;
}

PluginTypeID PluginIntegerType::getPluginTypeID()
{
    if (isSigned()) {
        switch (getWidth()) {
            case 1:
                return PluginTypeID::IntegerTy1ID;
            case 8:
                return PluginTypeID::IntegerTy8ID;
            case 16:
                return PluginTypeID::IntegerTy16ID;
            case 32:
                return PluginTypeID::IntegerTy32ID;
            case 64:
                return PluginTypeID::IntegerTy64ID;
            default:
                return PluginTypeID::UndefTyID;
        }
    }
    if (isUnsigned()) {
        switch (getWidth()) {
            case 1:
                return PluginTypeID::UIntegerTy1ID;
            case 8:
                return PluginTypeID::UIntegerTy8ID;
            case 16:
                return PluginTypeID::UIntegerTy16ID;
            case 32:
                return PluginTypeID::UIntegerTy32ID;
            case 64:
                return PluginTypeID::UIntegerTy64ID;
            default:
                return PluginTypeID::UndefTyID;
        }
    }
    return PluginTypeID::UndefTyID;
}

PluginIntegerType PluginIntegerType::get (MLIRContext *context, unsigned width,
    PluginIntegerType::SignednessSemantics signedness)
{
    return Base::get(context, width, signedness);
}

// ===----------------------------------------------------------------------===//
// Plugin Float Type
// ===----------------------------------------------------------------------===//

unsigned PluginFloatType::getWidth () const
{
    return getImpl()->width;
}

PluginTypeID PluginFloatType::getPluginTypeID()
{
    if (getWidth() == 32) {
        return PluginTypeID::FloatTyID;
    }
    if (getWidth() == 64) {
        return PluginTypeID::DoubleTyID;
    }
    return PluginTypeID::UndefTyID;
}

PluginFloatType PluginFloatType::get (MLIRContext *context, unsigned width)
{
    return Base::get(context, width);
}

// ===----------------------------------------------------------------------===//
// Plugin Boolean Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginBooleanType::getPluginTypeID()
{
    return PluginTypeID::BooleanTyID;
}

// ===----------------------------------------------------------------------===//
// Plugin Void Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginVoidType::getPluginTypeID()
{
    return PluginTypeID::VoidTyID;
}

// ===----------------------------------------------------------------------===//
// Plugin Undef Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginUndefType::getPluginTypeID()
{
    return PluginTypeID::UndefTyID;
}

// ===----------------------------------------------------------------------===//
// Plugin Pointer Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginPointerType::getPluginTypeID()
{
    return PluginTypeID::PointerTyID;
}

Type PluginPointerType::getElementType()
{
    return getImpl()->pointee;
}

unsigned PluginPointerType::isReadOnlyElem()
{
    return getImpl()->readOnlyPointee;
}

PluginPointerType PluginPointerType::get (MLIRContext *context, Type pointee, unsigned readOnlyPointee)
{
    return Base::get(context, pointee, readOnlyPointee);
}


// ===----------------------------------------------------------------------===//
// Plugin Array Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginArrayType::getPluginTypeID()
{
    return PluginTypeID::ArrayTyID;
}

bool PluginArrayType::isValidElementType(Type type)
{
    return !type.isa<PluginVoidType, PluginFunctionType, PluginUndefType>();
}

PluginArrayType PluginArrayType::get(MLIRContext *context, Type elementType, unsigned numElements)
{
    return Base::get(context, elementType, numElements);
}

Type PluginArrayType::getElementType()
{ 
    return getImpl()->elementType;
}

unsigned PluginArrayType::getNumElements()
{
    return getImpl()->numElements;
}

// ===----------------------------------------------------------------------===//
// Plugin Vector Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginVectorType::getPluginTypeID()
{
    return PluginTypeID::VectorTyID;
}

bool PluginVectorType::isValidElementType(Type type)
{
    return type.isa<PluginIntegerType, PluginFloatType>();
}

PluginVectorType PluginVectorType::get(MLIRContext *context, Type elementType, unsigned numElements)
{
    return Base::get(context, elementType, numElements);
}

Type PluginVectorType::getElementType()
{ 
    return getImpl()->elementType;
}

unsigned PluginVectorType::getNumElements()
{
    return getImpl()->numElements;
}

// ===----------------------------------------------------------------------===//
// Plugin Function Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginFunctionType::getPluginTypeID()
{
    return PluginTypeID::FunctionTyID;
}

bool PluginFunctionType::isValidArgumentType(Type type)
{
  return !type.isa<PluginVoidType, PluginFunctionType>();
}

bool PluginFunctionType::isValidResultType(Type type) {
  return !type.isa<PluginFunctionType>();
}

PluginFunctionType PluginFunctionType::get(MLIRContext *context, Type result, ArrayRef<Type> arguments)
{
    return Base::get(context, result, arguments);
}

Type PluginFunctionType::getReturnType()
{
    return getImpl()->resultType;
}

unsigned PluginFunctionType::getNumParams()
{
  return getImpl()->argumentTypes.size();
}

Type PluginFunctionType::getParamType(unsigned i) {
  return getImpl()->argumentTypes[i];
}

ArrayRef<Type> PluginFunctionType::getParams()
{
  return getImpl()->argumentTypes;
}

// ===----------------------------------------------------------------------===//
// Plugin Struct Type
// ===----------------------------------------------------------------------===//

PluginTypeID PluginStructType::getPluginTypeID()
{
    return PluginTypeID::StructTyID;
}

bool PluginStructType::isValidElementType(Type type) {
  return !type.isa<PluginVoidType, PluginFunctionType>();
}

PluginStructType PluginStructType::get(MLIRContext *context, StringRef name, ArrayRef<StringRef> elemNames)
{
    return Base::get(context, name, elemNames);
}

StringRef PluginStructType::getName()
{
  return getImpl()->name;
}

ArrayRef<StringRef> PluginStructType::getElementNames()
{
  return getImpl()->elemNames;
}