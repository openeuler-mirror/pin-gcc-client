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

#include "Translate/GimpleToPluginOps.h"
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

namespace PluginIR {
using namespace mlir::Plugin;
vector<FunctionOp> GimpleToPluginOps::GetAllFunction()
{
    cgraph_node *node = NULL;
    function *fn = NULL;
    vector<FunctionOp> functions;
    FOR_EACH_FUNCTION (node) {
        fn = DECL_STRUCT_FUNCTION(node->decl);
        if (fn == NULL)
            continue;
        int64_t id = reinterpret_cast<int64_t>(reinterpret_cast<void*>(fn));
        FunctionOp irFunc;
        mlir::StringRef funcName(function_name(fn));
        bool declaredInline = false;
        if (DECL_DECLARED_INLINE_P(fn->decl))
            declaredInline = true;
        auto location = builder.getUnknownLoc();
        irFunc = builder.create<FunctionOp>(location, id, funcName, declaredInline);
        functions.push_back(irFunc);
    }
    return functions;
}

vector<LocalDeclOp> GimpleToPluginOps::GetAllDecls(uint64_t funcID)
{
    function *fn = reinterpret_cast<function *>(funcID);
    vector<LocalDeclOp> decls;
    if (!vec_safe_is_empty(fn->local_decls)) {
        unsigned ix = 0;
        tree var = NULL_TREE;
        FOR_EACH_LOCAL_DECL (fn, ix, var) {
            uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(var));
            LocalDeclOp decl;
            if (TREE_CODE(var) != VAR_DECL || !DECL_NAME (var)) {
                continue;
            }
            const char* name = IDENTIFIER_POINTER(DECL_NAME (var));
            mlir::StringRef symName(name);
            auto location = builder.getUnknownLoc();
	    PluginTypeBase declType = typeTranslator.translateType((intptr_t)TREE_TYPE(var));
            int64_t typeID = typeTranslator.getPluginTypeId (declType);
            uint64_t typeWidth = typeTranslator.getBitWidth (declType);
            decl = builder.create<LocalDeclOp>(location, id, symName, typeID, typeWidth);
            decls.push_back(decl);
        }
    }
    return decls;
}

} // namespace PluginIR