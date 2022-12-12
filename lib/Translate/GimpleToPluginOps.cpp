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
#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallVector.h"

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
using namespace mlir;

namespace detail {
class BlockFromGimpleTranslatorImpl {
public:
    std::map<basic_block, Block*> blockMaps;
    /* Constructs a class creating types in the given MLIR context. */
    BlockFromGimpleTranslatorImpl(mlir::MLIRContext &context) : context(context) {}

private:
    /* The context in which MLIR types are created. */
    mlir::MLIRContext &context;
};

} // namespace detail

GimpleToPluginOps::GimpleToPluginOps (mlir::MLIRContext &context) :
 builder(&context), typeTranslator(context), bbTranslator(new detail::BlockFromGimpleTranslatorImpl(context))
{}

GimpleToPluginOps::~GimpleToPluginOps ()
{}

static IComparisonCode TranslateCmpCode(enum tree_code ccode)
{
    switch (ccode) {
        case LT_EXPR:
            return IComparisonCode::lt;
        case LE_EXPR:
            return IComparisonCode::le;
        case GT_EXPR:
            return IComparisonCode::gt;
        case GE_EXPR:
            return IComparisonCode::ge;
        case LTGT_EXPR:
            return IComparisonCode::ltgt;
        case EQ_EXPR:
            return IComparisonCode::eq;
        case NE_EXPR:
            return IComparisonCode::ne;
        default:
            printf("tcc_comparison: %d not suppoted!\n", ccode);
	        break;
    }
    return IComparisonCode::UNDEF;
}

static enum tree_code TranslateCmpCodeToTreeCode(IComparisonCode iCode)
{
    switch (iCode) {
        case IComparisonCode::lt:
            return LT_EXPR;
        case IComparisonCode::le:
            return LE_EXPR;
        case IComparisonCode::gt:
            return GT_EXPR;
        case IComparisonCode::ge:
            return GE_EXPR;
        case IComparisonCode::ltgt:
            return LTGT_EXPR;
        case IComparisonCode::eq:
            return EQ_EXPR;
        case IComparisonCode::ne:
            return NE_EXPR;
        default:
            printf("tcc_comparison not suppoted!\n");
	        break;
    }
    return LT_EXPR;
}

static IExprCode TranslateExprCode(enum tree_code ccode)
{
    switch (ccode) {
        case PLUS_EXPR:
            return IExprCode::Plus;
        case MINUS_EXPR:
            return IExprCode::Minus;
        case NOP_EXPR:
            return IExprCode::Nop;
        default:
            // printf("tcc_binary: %d not suppoted!\n", ccode);
	        break;
    }
    return IExprCode::UNDEF;
}

static StringRef GimpleCodeToOperationName(enum gimple_code tcode)
{
    StringRef ret;
    switch (tcode) {
        case GIMPLE_PHI: {
            ret = PhiOp::getOperationName();
            break;
        }
        case GIMPLE_ASSIGN: {
            ret = AssignOp::getOperationName();
            break;
        }
        case GIMPLE_CALL: {
            ret = CallOp::getOperationName();
            break;
        }
        case GIMPLE_COND: {
            ret = CondOp::getOperationName();
            break;
        }
        default: {
            ret = BaseOp::getOperationName();
            break;
        }
    }
    return ret;
}

uint64_t GimpleToPluginOps::CreateBlock(uint64_t funcAddr, uint64_t bbAddr)
{
    basic_block address = reinterpret_cast<basic_block>(bbAddr);
    function *fn = reinterpret_cast<function *>(funcAddr);
    push_cfun(fn);
    uint64_t ret = reinterpret_cast<uint64_t>(create_empty_bb(address));
    pop_cfun();
    return ret;
}

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
        FunctionOp irFunc = BuildFunctionOp(id);
        functions.push_back(irFunc);
        builder.setInsertionPointAfter(irFunc.getOperation());
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

FunctionOp GimpleToPluginOps::BuildFunctionOp(uint64_t functionId)
{
    function *fn = reinterpret_cast<function*>(functionId);
    mlir::StringRef funcName(function_name(fn));
    bool declaredInline = false;
    if (DECL_DECLARED_INLINE_P(fn->decl))
        declaredInline = true;
    auto location = builder.getUnknownLoc();
    FunctionOp retOp = builder.create<FunctionOp>(location, functionId,
                                        funcName, declaredInline);
    auto& fr = retOp.bodyRegion();
    if (!ProcessBasicBlock((intptr_t)ENTRY_BLOCK_PTR_FOR_FN(fn), fr)) {
        // handle error
        return retOp;
    }
    return retOp;
}

Operation *GimpleToPluginOps::BuildOperation(uint64_t id)
{
    gimple *stmt = reinterpret_cast<gimple*>(id);
    Operation *ret = nullptr;
    switch (gimple_code(stmt)) {
        case GIMPLE_PHI: {
            PhiOp phiOp = BuildPhiOp(id);
            ret = phiOp.getOperation();
            break;
        }
        case GIMPLE_ASSIGN: {
            AssignOp assignOp = BuildAssignOp(id);
            ret = assignOp.getOperation();
            break;
        }
        case GIMPLE_CALL: {
            CallOp callOp = BuildCallOp(id);
            ret = callOp.getOperation();
            break;
        }
        case GIMPLE_COND: {
            assert(EDGE_COUNT (stmt->bb->succs) == 2);
            Block* trueBlock = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 0)->dest];
            Block* falseBlock = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 1)->dest];
            CondOp condOp = BuildCondOp(id, (uint64_t)stmt->bb,
                                        trueBlock, falseBlock,
                                        (uint64_t)EDGE_SUCC(stmt->bb, 0)->dest,
                                        (uint64_t)EDGE_SUCC(stmt->bb, 1)->dest);
            ret = condOp.getOperation();
            break;
        }
        default: {
            BaseOp baseOp = builder.create<BaseOp>(
                    builder.getUnknownLoc(), id, BaseOp::getOperationName());
            ret = baseOp.getOperation();
            break;
        }
    }
    return ret;
}

PhiOp GimpleToPluginOps::BuildPhiOp(uint64_t gphiId)
{
    gphi *stmt = reinterpret_cast<gphi*>(gphiId);
    llvm::SmallVector<Value, 4> ops;
    ops.reserve(gimple_phi_num_args(stmt));
    for (unsigned i = 0; i < gimple_phi_num_args(stmt); i++) {
        tree *argTree = gimple_phi_arg_def_ptr(stmt, i);
        uint64_t argId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(argTree));
        Value arg = TreeToValue(argId);
        ops.push_back(arg);
    }
    tree returnType = TREE_TYPE(gimple_phi_result(stmt));
    PluginTypeBase rPluginType = typeTranslator.translateType((intptr_t)returnType);
    uint32_t capacity = gimple_phi_capacity(stmt);
    uint32_t nArgs = gimple_phi_num_args(stmt);
    PhiOp ret = builder.create<PhiOp>(builder.getUnknownLoc(),
                                      gphiId, capacity, nArgs, ops, rPluginType);
    return ret;
}

CallOp GimpleToPluginOps::BuildCallOp(uint64_t gcallId)
{
    gcall *stmt = reinterpret_cast<gcall*>(gcallId);
    tree fndecl = gimple_call_fndecl(stmt);
    if (fndecl == NULL_TREE || DECL_NAME(fndecl) == NULL_TREE) {
        return nullptr;
    }
    llvm::SmallVector<Value, 4> ops;
    ops.reserve(gimple_call_num_args(stmt));
    for (unsigned i = 0; i < gimple_call_num_args(stmt); i++) {
        tree *callArg = gimple_call_arg_ptr(stmt, i);
        uint64_t argId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(callArg));
        Value arg = TreeToValue(argId);
        ops.push_back(arg);
    }
    StringRef callName(IDENTIFIER_POINTER(DECL_NAME(fndecl)));
    tree returnType = gimple_call_return_type(stmt);
    PluginTypeBase rPluginType = typeTranslator.translateType((intptr_t)returnType);
    CallOp ret = builder.create<CallOp>(builder.getUnknownLoc(), gcallId, callName, ops);
    return ret;
}

bool GimpleToPluginOps::SetGimpleCallLHS(uint64_t callId, uint64_t lhsId)
{
    gcall *stmt = reinterpret_cast<gcall*>(callId);
    tree *lhs = reinterpret_cast<tree*>(callId);
    gimple_call_set_lhs (stmt, *lhs);
    return true;
}

uint64_t GimpleToPluginOps::CreateGcond(IComparisonCode iCode,
                                        uint64_t lhsId, uint64_t rhsId)
{
    tree *lhs = reinterpret_cast<tree*>(lhsId);
    tree *rhs = reinterpret_cast<tree*>(rhsId);
    gcond *ret = gimple_build_cond (TranslateCmpCodeToTreeCode(iCode),
                                    *lhs, *rhs, NULL_TREE, NULL_TREE);
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(ret));
}

CondOp GimpleToPluginOps::BuildCondOp(uint64_t gcondId, uint64_t address,
                                      Block* b1, Block* b2, uint64_t tbaddr,
                                      uint64_t fbaddr)
{
    gcond *stmt = reinterpret_cast<gcond*>(gcondId);
    tree *lhsPtr = gimple_cond_lhs_ptr(stmt);
    uint64_t lhsId = reinterpret_cast<uint64_t>(
        reinterpret_cast<void*>(lhsPtr));
    Value LHS = TreeToValue(lhsId);
    tree *rhsPtr = gimple_cond_rhs_ptr(stmt);
    uint64_t rhsId = reinterpret_cast<uint64_t>(
        reinterpret_cast<void*>(rhsPtr));
    Value RHS = TreeToValue(rhsId);
    Value trueLabel = nullptr;
    Value falseLabel = nullptr;
    IComparisonCode iCode = TranslateCmpCode(gimple_cond_code(stmt));
    CondOp ret = builder.create<CondOp>(builder.getUnknownLoc(), gcondId,
                                        address, iCode, LHS, RHS, b1, b2,
                                        tbaddr, fbaddr, trueLabel, falseLabel);
    return ret;
}

AssignOp GimpleToPluginOps::BuildAssignOp(uint64_t gassignId)
{
    gassign *stmt = reinterpret_cast<gassign*>(gassignId);
    llvm::SmallVector<Value, 4> ops;
    ops.reserve(gimple_num_ops(stmt));
    uint64_t lhsId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_assign_lhs_ptr(stmt)));
    ops.push_back(TreeToValue(lhsId));
    uint64_t rhs1Id = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_assign_rhs1_ptr(stmt)));
    ops.push_back(TreeToValue(rhs1Id));
    if (gimple_assign_rhs2(stmt) != NULL_TREE) {
        uint64_t rhs2Id = reinterpret_cast<uint64_t>(
                reinterpret_cast<void*>(gimple_assign_rhs2_ptr(stmt)));
        ops.push_back(TreeToValue(rhs2Id));
    }
    if (gimple_assign_rhs3(stmt) != NULL_TREE) {
        uint64_t rhs3Id = reinterpret_cast<uint64_t>(
                reinterpret_cast<void*>(gimple_assign_rhs3_ptr(stmt)));
        ops.push_back(TreeToValue(rhs3Id));
    }
    IExprCode iCode = TranslateExprCode(gimple_assign_rhs_code(stmt));
    mlir::Type returnType = nullptr;
    AssignOp ret = builder.create<AssignOp>(
            builder.getUnknownLoc(), gassignId, iCode, ops, returnType);
    return ret;
}

Value GimpleToPluginOps::GetGphiResult(uint64_t id)
{
    gphi *stmt = reinterpret_cast<gphi*>(id);
    tree ret = gimple_phi_result(stmt);
    uint64_t retId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(ret));
    return TreeToValue(retId);
}

Value GimpleToPluginOps::TreeToValue(uint64_t treeId)
{
    tree *t = reinterpret_cast<tree*>(treeId);
    tree treeType = TREE_TYPE(*t);
    PluginTypeBase rPluginType = typeTranslator.translateType((intptr_t)treeType);
    switch (TREE_CODE(*t)) {
        case MEM_REF : {
            break;
        }
        case INTEGER_CST : {
            break;
        }
        case SSA_NAME : {
            break;
        }
        default:
            break;
    }
    mlir::Value opValue = builder.create<PlaceholderOp>(
            builder.getUnknownLoc(), treeId,
            IDefineCode::UNDEF, rPluginType);
    return opValue;
}

bool GimpleToPluginOps::ProcessGimpleStmt(intptr_t bbPtr, Region& rg)
{
    bool putTerminator = false;
    basic_block bb = reinterpret_cast<basic_block>(bbPtr);
    for (gphi_iterator si = gsi_start_phis (bb); !gsi_end_p (si); gsi_next (&si)) {
        gphi *p = si.phi ();
        uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(p));
        BuildPhiOp(id); // FIXME: Check result. 
    }

    for (gimple_stmt_iterator si = gsi_start_bb (bb); !gsi_end_p (si); gsi_next (&si)) {
        gimple *stmt = gsi_stmt (si);
        uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmt));
        if (BuildOperation(id) == nullptr) {
            printf("ERROR: BuildOperation!");
        }
        if(gimple_code(stmt) == GIMPLE_COND) {
            putTerminator = true;
        }
    }

    if (!putTerminator) {
        // Process fallthrough, todo: process goto
        if (EDGE_COUNT (bb->succs) == 1) {
            builder.create<FallThroughOp>(builder.getUnknownLoc(), (uint64_t)bb,
                            bbTranslator->blockMaps[EDGE_SUCC(bb, 0)->dest],
                            (uint64_t)(EDGE_SUCC(bb, 0)->dest));
        } else if (!EDGE_COUNT (bb->succs)) {
            // Process other condition, such as return
            builder.create<RetOp>(builder.getUnknownLoc(), (uint64_t)bb);
        } else {
            // Should unreachable;
            assert(false);
        }
    }

    return true;
}

bool GimpleToPluginOps::ProcessBasicBlock(intptr_t bbPtr, Region& rg)
{
    basic_block bb = reinterpret_cast<basic_block>(bbPtr);
    // handled, skip process
    if (bbTranslator->blockMaps.find(bb) != bbTranslator->blockMaps.end()) {
        return true;
    }
    // fprintf(stderr,"processing bb[%d]\n", bb->index);

    // create basic block
    Block* block = builder.createBlock(&rg, rg.begin());
    bbTranslator->blockMaps.insert({bb, block});
    // todo process func return type
    // todo isDeclaration

    // process succ
    for (unsigned int i = 0; i < EDGE_COUNT (bb->succs); i++) {
        // fprintf(stderr,"-->[%d]\n", EDGE_SUCC(bb, i)->dest->index);
        if (!ProcessBasicBlock((intptr_t)(EDGE_SUCC(bb, i)->dest), rg)) {
            return false;
        }
    }
    // process each stmt
    builder.setInsertionPointToStart(block);
    if (!ProcessGimpleStmt(bbPtr, rg)) {
        return false;
    }
    // block->dump();
    // fprintf(stderr, "[bb%d] succ: %d\n", bb->index,block->getNumSuccessors());
    return true;
}

} // namespace PluginIR