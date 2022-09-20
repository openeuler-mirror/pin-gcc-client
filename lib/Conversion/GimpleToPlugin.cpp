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

   Author: Mingchuan Wu and Yancheng Li
   Create: 2022-08-18
   Description:
    This file contains the implementation of the GimpleToPlugin class.
*/

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

#include "Conversion/GimpleToPlugin.h"

namespace Plugin_IR {
vector<Operation> GimpleToPlugin::GetAllFunction()
{
    cgraph_node *node = NULL;
    function *fn = NULL;
    vector<Operation> functions;
    FOR_EACH_FUNCTION (node) {
        fn = DECL_STRUCT_FUNCTION(node->decl);
        if (fn == NULL)
            continue;
        Operation irFunc(OP_FUNCTION);
        irFunc.SetID(
            reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(fn)));
        irFunc.AddAttribute("name", function_name(fn));
        if (DECL_DECLARED_INLINE_P(fn->decl))
            irFunc.AddAttribute("declaredInline", "1");
        else
            irFunc.AddAttribute("declaredInline", "0");

        // Transform LOCAL_DECL.
        if (fn->local_decls) {
            string localDecl;
            tree var;
            unsigned i;
            FOR_EACH_LOCAL_DECL (fn, i, var) {
                uintptr_t declID = reinterpret_cast<uintptr_t>(
                    reinterpret_cast<void*>(var));
                if (!localDecl.empty())
                    localDecl += ",";
                localDecl += std::to_string(declID);
            }
            irFunc.AddAttribute("localDecl", localDecl);
        } else {
            irFunc.AddAttribute("localDecl", "");
        }
        functions.push_back(irFunc);
    }
    return functions;
}

Decl GimpleToPlugin::GetDeclByID(uintptr_t id)
{
    tree var = reinterpret_cast<tree>(id);
    Decl irDecl;
    if (DECL_NAME(var) == NULL)
        return irDecl;
    if (TREE_CODE (var) == VAR_DECL) {
        irDecl.SetDeclCode(DC_VAR);
        irDecl.SetID(id);
        irDecl.AddAttribute("name", IDENTIFIER_POINTER(DECL_NAME(var)));
        uintptr_t typeID = reinterpret_cast<uintptr_t>(
            reinterpret_cast<void*>(TREE_TYPE(var)));
        irDecl.SetType(GetTypeByID(typeID));
    }
    return irDecl;
}

static const char *GetTypeName(tree type)
{
    const char *tname = NULL;
    if (type == NULL)
        return nullptr;
    if (TYPE_NAME (type) != NULL) {
        if (TREE_CODE (TYPE_NAME (type)) == IDENTIFIER_NODE) {
            tname = IDENTIFIER_POINTER (TYPE_NAME (type));
        } else if (DECL_NAME (TYPE_NAME (type)) != NULL) {
            tname = IDENTIFIER_POINTER (DECL_NAME (TYPE_NAME (type)));
        }
    }
    return tname;
}

static const TypeCode TransformTypeCode(tree type)
{
    TypeCode retCode = TC_UNDEF;
    switch (TREE_CODE(type)) {
        case BOOLEAN_TYPE:
            retCode = TC_BOOL;
            break;
        case VOID_TYPE:
            retCode = TC_VOID;
            break;
        case INTEGER_TYPE: {
            unsigned precision = TYPE_PRECISION (type);
            if (TYPE_UNSIGNED (type) == 1) {
                if (precision == 1)
                    retCode = TC_U1;
                else if (precision == 8) // size of 1 byte
                    retCode = TC_U8;
                else if (precision == 16) // size of 2 byte
                    retCode = TC_U16;
                else if (precision == 32) // size of 4 byte
                    retCode = TC_U32;
                else if (precision == 64) // size of 8 byte
                    retCode = TC_U64;
            } else {
                if (precision == 1)
                    retCode = TC_I1;
                else if (precision == 8) // size of 1 byte
                    retCode = TC_I8;
                else if (precision == 16) // size of 2 byte
                    retCode = TC_I16;
                else if (precision == 32) // size of 4 byte
                    retCode = TC_I32;
                else if (precision == 64) // size of 8 byte
                    retCode = TC_I64;
            }
            break;
        }
        case REAL_TYPE: {
            unsigned precision = TYPE_PRECISION (type);
            if (precision == 16) // float with 16 bits
                retCode = TC_FP16;
            else if (precision == 32) // float with 32 bits
                retCode = TC_FP32;
            else if (precision == 64) // float with 64 bits
                retCode = TC_FP64;
            else if (precision == 80) // float with 80 bits
                retCode = TC_FP80;
        }
        default:
            break;
    }
    return retCode;
}

Type GimpleToPlugin::GetTypeByID(uintptr_t id)
{
    tree type = reinterpret_cast<tree>(id);
    Type irType;
    const char *typeName = GetTypeName(type);
    if (typeName == NULL)
        return irType;
    irType.SetID(id);
    irType.SetTypeCode(TransformTypeCode(type));
    uint8_t tq = TQ_UNDEF;
    if (TREE_THIS_VOLATILE (type))
        tq |= TQ_VOLATILE;
    if (TREE_CONSTANT (type))
        tq |= TQ_CONST;
    irType.SetTQual(tq);
    irType.AddAttribute("name", typeName);
    return irType;
}
} // namespace Plugin_IR