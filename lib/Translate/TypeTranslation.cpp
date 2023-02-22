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
#include "print-tree.h"
#include "stor-layout.h"


namespace PluginIR {
using namespace mlir;
namespace Detail {
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

    unsigned getDomainIndex (tree type)
    {
        return tree_to_shwi(TYPE_MAX_VALUE(TYPE_DOMAIN(type)))+1;
    }

    llvm::SmallVector<Type> getArgsType (tree type)
    {
        tree parmlist = TYPE_ARG_TYPES (type);
        tree parmtype;
        llvm::SmallVector<Type> typelist;
        for (; parmlist; parmlist = TREE_CHAIN (parmlist))
        {
            parmtype = TREE_VALUE (parmlist);
            typelist.push_back(translatePrimitiveType(parmtype));
        }
        return typelist;
    }

    const char *getTypeName (tree type)
    {
        const char *tname = NULL;

        if (type == NULL)
        {
            return NULL;
        }

        if (TYPE_NAME (type) != NULL)
        {
            if (TREE_CODE (TYPE_NAME (type)) == IDENTIFIER_NODE)
	        {
	            tname = IDENTIFIER_POINTER (TYPE_NAME (type));
	        }
            else if (DECL_NAME (TYPE_NAME (type)) != NULL)
	        {
	            tname = IDENTIFIER_POINTER (DECL_NAME (TYPE_NAME (type)));
	        }
        }
        return tname;
    }

    llvm::SmallVector<Type> getElemType(tree type)
    {
        llvm::SmallVector<Type> typelist;
        tree parmtype;
        for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
        {
            if (TREE_CODE (field) == FIELD_DECL)
            {
                parmtype = TREE_TYPE(field);
                typelist.push_back(translatePrimitiveType(parmtype));
            }
        }
        return typelist;
    }
    
    llvm::SmallVector<StringRef> getElemNames(tree type)
    {
        llvm::SmallVector<StringRef> names;
        StringRef name;
        for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
        {
            if (TREE_CODE (field) == FIELD_DECL)
            {
                name = IDENTIFIER_POINTER ( DECL_NAME(field));
                names.push_back(name);
            }
        }
        return names;
    }

    /* Translates the given primitive, i.e. non-parametric in MLIR nomenclature,
       type. */
    PluginTypeBase translatePrimitiveType (tree type)
    {
        if (TREE_CODE(type) == INTEGER_TYPE)
            return PluginIntegerType::get(&context, getBitWidth(type),
                isUnsigned(type) ? PluginIntegerType::Unsigned : PluginIntegerType::Signed);
        if (TREE_CODE(type) == REAL_TYPE)
            return PluginFloatType::get(&context, getBitWidth(type));
        if (TREE_CODE(type) == BOOLEAN_TYPE)
            return PluginBooleanType::get(&context);
        if (TREE_CODE(type) == VOID_TYPE)
            return PluginVoidType::get(&context);
        if (TREE_CODE(type) == POINTER_TYPE)
            return PluginPointerType::get(&context, translatePrimitiveType(TREE_TYPE(type)),
                TYPE_READONLY(TREE_TYPE(type)) ? 1 : 0);
        if (TREE_CODE(type) == ARRAY_TYPE)
            return PluginArrayType::get(&context,translatePrimitiveType(TREE_TYPE(type)), getDomainIndex(type));
        if (TREE_CODE(type) == FUNCTION_TYPE) {
            llvm::SmallVector<Type> argsType = getArgsType(type);
            return PluginFunctionType::get(&context, translatePrimitiveType(TREE_TYPE(type)),argsType);
        }
        if (TREE_CODE(type) == RECORD_TYPE) {
            return PluginStructType::get(&context, getTypeName(type), getElemType(type), getElemNames(type));
        }
        return PluginUndefType::get(&context);
    }

    /* The context in which MLIR types are created. */
    mlir::MLIRContext &context;
}; // class TypeFromPluginIRTranslatorImpl

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
    unsigned getBitWidth(PluginIntegerType type)
    {
        return type.getWidth();
    }

    bool isUnsigned(PluginIntegerType type)
    {
        if (type.isUnsigned()) {
            return true;
        }
        return false;
    }

    auto_vec<tree> getParamsType(PluginFunctionType Ty)
    {
        auto_vec<tree> paramTypes;
        ArrayRef<Type> ArgsTypes = Ty.getParams();
        for (auto ty :ArgsTypes) {
            paramTypes.safe_push(translatePrimitiveType(ty.dyn_cast<PluginTypeBase>()));
        }
        return paramTypes;
    }

    tree translatePrimitiveType(PluginTypeBase type)
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
            unsigned elmConst = Ty.isReadOnlyElem();
            auto ty = elmType.dyn_cast<PluginTypeBase>();
            tree elmTy = translatePrimitiveType(ty);
            TYPE_READONLY(elmTy) = elmConst ? 1 : 0;
            return build_pointer_type(elmTy);
        }
        if (auto Ty = type.dyn_cast<PluginArrayType>()) {
            mlir::Type elmType = Ty.getElementType();
            auto ty = elmType.dyn_cast<PluginTypeBase>();
            tree elmTy = translatePrimitiveType(ty);
            unsigned elmNum = Ty.getNumElements();
            tree index = build_index_type (size_int (elmNum));
            return build_array_type(elmTy, index);
        }
        if (auto Ty = type.dyn_cast<PluginFunctionType>()) {
            Type resultType = Ty.getReturnType();
            tree returnType = translatePrimitiveType(resultType.dyn_cast<PluginTypeBase>());
            auto_vec<tree> paramTypes = getParamsType(Ty);
            return build_function_type_array(returnType, paramTypes.length (), paramTypes.address ());
        }
        if (auto Ty = type.dyn_cast<PluginStructType>()) {
            ArrayRef<Type> elemTypes = Ty.getBody();
            ArrayRef<StringRef> elemNames = Ty.getElementNames();
            StringRef tyName = Ty.getName();
            unsigned fieldSize = elemNames.size();

            tree fields[fieldSize];
            tree ret;
            unsigned i;

            ret = make_node (RECORD_TYPE);
            for (i = 0; i < fieldSize; i++)
            {
                mlir::Type elemTy = elemTypes[i];
                auto ty = elemTy.dyn_cast<PluginTypeBase>();
                tree elmType = translatePrimitiveType(ty);
                fields[i] = build_decl (UNKNOWN_LOCATION, FIELD_DECL, get_identifier (elemNames[i].str().c_str()), elmType);
                DECL_CONTEXT (fields[i]) = ret;
                if (i) DECL_CHAIN (fields[i - 1]) = fields[i];
            }
            tree typeDecl = build_decl (input_location, TYPE_DECL, get_identifier (tyName.str().c_str()), ret);
            DECL_ARTIFICIAL (typeDecl) = 1;
            TYPE_FIELDS (ret) = fields[0];
            TYPE_NAME (ret) = typeDecl;
            layout_type (ret);
            return ret;
        }
        return NULL;
    }
}; // class TypeToPluginIRTranslatorImpl

} // namespace Detail
} // namespace PluginIR

PluginIR::TypeFromPluginIRTranslator::TypeFromPluginIRTranslator(mlir::MLIRContext &context)
    : impl(new Detail::TypeFromPluginIRTranslatorImpl(context)) {}

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
    : impl(new Detail::TypeToPluginIRTranslatorImpl()) {}

PluginIR::TypeToPluginIRTranslator::~TypeToPluginIRTranslator() {}

uintptr_t PluginIR::TypeToPluginIRTranslator::translateType(PluginIR::PluginTypeBase type)
{
    return impl->translateType(type);
}