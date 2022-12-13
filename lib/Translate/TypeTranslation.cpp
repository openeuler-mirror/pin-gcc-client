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
    Type translation between MLIR Plugin & Plugin IR.
*/

#include "Translate/TypeTranslation.h"
#include "Dialect/PluginTypes.h"

#include <cstdio>

#include "gcc-plugin.h"
#include "plugin-version.h"
#include "tree-pass.h"
#include "context.h"
#include "coretypes.h"
#include "tree.h"
#include "cgraph.h"
#include "diagnostic-core.h"
#include "function.h"
#include "basic-block.h"
#include "gimple.h"
#include "vec.h"
#include "tree-pretty-print.h"
#include "gimple-pretty-print.h"
#include "gimple-iterator.h"
#include "gimple-walk.h"
#include "cfg.h"
#include "ssa.h"
#include "output.h"
#include "langhooks.h"

using namespace mlir;

namespace PluginIR {
namespace detail {
/* Support for translating Plugin IR types to MLIR Plugin dialect types. */
class TypeFromPluginIRTranslatorImpl {
public:
    /* Constructs a class creating types in the given MLIR context. */
    TypeFromPluginIRTranslatorImpl(mlir::MLIRContext &context) : context(context) {}

    PluginTypeBase translateType (uintptr_t id)
    {
        tree node = reinterpret_cast<tree>(id);
        PluginTypeBase type = translatePrimitiveType (node);
        return type;
    }

private:
    unsigned getBitWidth (tree type)
    {
        unsigned precision = TYPE_PRECISION (type);
        return precision;
    }

    bool isUnsigned (tree type)
    {
        if (TYPE_UNSIGNED (type))
            return true;
        return false;
    }

    /* Translates the given primitive, i.e. non-parametric in MLIR nomenclature,
       type. */
    PluginTypeBase translatePrimitiveType (tree type)
    {
        if (TREE_CODE(type) == INTEGER_TYPE)
            return PluginIntegerType::get(&context, getBitWidth(type), isUnsigned(type) ? PluginIntegerType::Unsigned : PluginIntegerType::Signed);
        if (TREE_CODE(type) == REAL_TYPE)
            return PluginFloatType::get(&context, getBitWidth(type));
        if (TREE_CODE(type) == BOOLEAN_TYPE)
            return PluginBooleanType::get(&context);
        if (TREE_CODE(type) == VOID_TYPE)
            return PluginVoidType::get(&context);
        if (TREE_CODE(type) == POINTER_TYPE)
            return PluginPointerType::get(&context, translatePrimitiveType(TREE_TYPE(type)));
        return PluginUndefType::get(&context);
    }

    /* The context in which MLIR types are created. */
    mlir::MLIRContext &context;
};

/* Support for translating MLIR Plugin dialect types to Plugin IR types . */
class TypeToPluginIRTranslatorImpl {
public:
    TypeToPluginIRTranslatorImpl() {}

    uintptr_t translateType (PluginTypeBase type)
    {
        tree node = translatePrimitiveType (type);
        assert(node!=NULL);
        return reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(node));
    }

private:
    unsigned getBitWidth (PluginIntegerType type)
    {
        return type.getWidth();
    }

    bool isUnsigned (PluginIntegerType type)
    {
        if(type.isUnsigned())
            return true;
        return false;
    }

    tree translatePrimitiveType (PluginTypeBase type)
    {
        if (auto Ty = type.dyn_cast<PluginIntegerType>()) {
            if (isUnsigned(Ty)) {
                switch (getBitWidth(Ty)) {
                    case 8:
                        return unsigned_char_type_node;
                    case 16:
                        return short_unsigned_type_node;
                    case 32:
                        return unsigned_type_node;
                    case 64:
                        return long_unsigned_type_node;
                    default:
                        return NULL;
                }
            } else {
                switch (getBitWidth(Ty)) {
                    case 8:
                        return signed_char_type_node;
                    case 16:
                        return short_integer_type_node;
                    case 32:
                        return integer_type_node;
                    case 64:
                        return long_integer_type_node;
                    default:
                        return NULL;
                }
            }
        }
        if (auto Ty = type.dyn_cast<PluginFloatType>()) {
            if (Ty.getWidth() == 32) {
                return float_type_node;
            } else if (Ty.getWidth() == 64) {
                return double_type_node;
            }
            return NULL;
        }
        if (auto Ty = type.dyn_cast<PluginBooleanType>()) {
            return boolean_type_node;
        }
        if (auto Ty = type.dyn_cast<PluginVoidType>()) {
            return void_type_node;
        }
        if (auto Ty = type.dyn_cast<PluginPointerType>()) {
            mlir::Type elmType = Ty.getElementType();
            auto ty = elmType.dyn_cast<PluginTypeBase>();
            tree elmTy = translatePrimitiveType(ty);
            return build_pointer_type(elmTy);
        }
        return NULL;
    }

};

} // namespace detail
} // namespace PluginIR

PluginIR::TypeFromPluginIRTranslator::TypeFromPluginIRTranslator(mlir::MLIRContext &context)
    : impl(new detail::TypeFromPluginIRTranslatorImpl(context)) {}

PluginIR::TypeFromPluginIRTranslator::~TypeFromPluginIRTranslator() {}

PluginIR::PluginTypeBase PluginIR::TypeFromPluginIRTranslator::translateType(uintptr_t id)
{
    return impl->translateType(id);
}

PluginIR::PluginTypeID PluginIR::TypeFromPluginIRTranslator::getPluginTypeId(PluginTypeBase type)
{
    return type.getPluginTypeID ();
}

uint64_t PluginIR::TypeFromPluginIRTranslator::getBitWidth(PluginTypeBase type)
{
    return type.getPluginIntOrFloatBitWidth ();
}


PluginIR::TypeToPluginIRTranslator::TypeToPluginIRTranslator()
    : impl(new detail::TypeToPluginIRTranslatorImpl()) {}

PluginIR::TypeToPluginIRTranslator::~TypeToPluginIRTranslator() {}

uintptr_t PluginIR::TypeToPluginIRTranslator::translateType(PluginIR::PluginTypeBase type)
{
    return impl->translateType(type);
}