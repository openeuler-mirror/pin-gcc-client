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
#include "cfgloop.h"
#include "tree-cfg.h"

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

vector<LoopOp> GimpleToPluginOps::GetAllLoops(uint64_t funcID)
{
    function *fn = reinterpret_cast<function *>(funcID);
    push_cfun(fn);
    vector<LoopOp> loops;
    enum li_flags LI = LI_FROM_INNERMOST;
    class loop *loop;
    FOR_EACH_LOOP(loop, LI) {
        uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop));
        LoopOp pluginLoop;
        if (!id) {
            continue;
        }
        auto location = builder.getUnknownLoc();
        uint32_t index = (uint32_t)loop->num;
        uint64_t innerLoopId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop->inner));
        uint64_t outerLoopId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop_outer(loop)));
        uint32_t numBlock = loop->num_nodes;
        pluginLoop = builder.create<LoopOp>(location, id, index, innerLoopId, outerLoopId, numBlock);
        loops.push_back(pluginLoop);
    }
    pop_cfun();
    return loops;
}

LoopOp GimpleToPluginOps::GetLoop(uint64_t loopID)
{
    assert(loopID);
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    uint64_t id = loopID;
    LoopOp pluginLoop;
    auto location = builder.getUnknownLoc();
    uint32_t index = (uint32_t)loop->num;
    uint64_t innerLoopId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop->inner));
    uint64_t outerLoopId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop_outer(loop)));
    uint32_t numBlock = loop->num_nodes;
    pluginLoop = builder.create<LoopOp>(location, id, index, innerLoopId, outerLoopId, numBlock);
    return pluginLoop;
}

bool GimpleToPluginOps::IsBlockInside(uint64_t loopID, uint64_t blockID)
{
    assert(loopID && blockID);
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    basic_block bb = reinterpret_cast<basic_block>(reinterpret_cast<void*>(blockID));
    return flow_bb_inside_loop_p(loop, bb);
}

vector<uint64_t> GimpleToPluginOps::GetBlocksInLoop(uint64_t loopID)
{
    assert(loopID);
    vector<uint64_t> blocks;
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    basic_block *bbs = get_loop_body_in_dom_order(loop);
    for (unsigned i = 0; i < loop->num_nodes; i++) {
        uint64_t blockId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(bbs[i]));
        blocks.push_back(blockId);
    }
    return blocks;
}

uint64_t GimpleToPluginOps::AllocateNewLoop(void)
{
    class loop *loop = alloc_loop();
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop));
}

void GimpleToPluginOps::DeleteLoop(uint64_t loopID)
{
    class loop *loop = reinterpret_cast<class loop *>(reinterpret_cast<void *>(loopID));
    delete_loop(loop);
}

void GimpleToPluginOps::AddLoop(uint64_t loopID, uint64_t outerID, uint64_t funcID)
{
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    class loop *outer = reinterpret_cast<class loop *>(outerID);
    function *fn = reinterpret_cast<function*>(funcID);
    push_cfun(fn);
    add_loop(loop, outer);
    pop_cfun();
}

uint64_t GimpleToPluginOps::GetHeader(uint64_t loopID)
{
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    basic_block header = loop->header;
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(header));
}

uint64_t GimpleToPluginOps::GetLatch(uint64_t loopID)
{
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    basic_block latch = loop->latch;
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(latch));
}

vector<std::pair<uint64_t, uint64_t> > GimpleToPluginOps::GetLoopExits(uint64_t loopID)
{
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    vec<edge> exit_edges = get_loop_exit_edges(loop);
    edge e;
    unsigned i = 0;
    vector<std::pair<uint64_t, uint64_t> > res;
    FOR_EACH_VEC_ELT(exit_edges, i, e) {
        res.push_back(std::make_pair((uint64_t)e->src, (uint64_t)e->dest));
    }
    return res;
}

std::pair<uint64_t, uint64_t> GimpleToPluginOps::GetLoopSingleExit(uint64_t loopID)
{
    class loop *loop = reinterpret_cast<class loop *>(loopID);
    edge e = single_exit(loop);
    std::pair<uint64_t, uint64_t> res;
    res.first = e ? (uint64_t)e->src : 0;
    res.second = e ? (uint64_t)e->dest : 0;
    return res;
}

LoopOp GimpleToPluginOps::GetBlockLoopFather(uint64_t blockID)
{
    basic_block bb = reinterpret_cast<basic_block>(reinterpret_cast<void*>(blockID));
    class loop *loop = bb->loop_father;
    LoopOp pluginLoop;
    auto location = builder.getUnknownLoc();
    uint64_t id = reinterpret_cast<uint64_t>(loop);
    uint32_t index = (uint32_t)loop->num;
    uint64_t innerLoopId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop->inner));
    uint64_t outerLoopId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(loop_outer(loop)));
    uint32_t numBlock = loop->num_nodes;
    pluginLoop = builder.create<LoopOp>(location, id, index, innerLoopId, outerLoopId, numBlock);
    return pluginLoop;
}

} // namespace PluginIR