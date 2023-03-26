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
#include "tree-into-ssa.h"
#include "dominance.h"
#include "print-tree.h"
#include "stor-layout.h"

namespace PluginIR {
using namespace mlir::Plugin;
using namespace mlir;

namespace Detail {
class BlockFromGimpleTranslatorImpl {
public:
    std::map<basic_block, Block*> blockMaps;
    /* Constructs a class creating types in the given MLIR context. */
    BlockFromGimpleTranslatorImpl(mlir::MLIRContext &context) : context(context) {}

private:
    /* The context in which MLIR types are created. */
    mlir::MLIRContext &context;
};

} // namespace Detail

GimpleToPluginOps::GimpleToPluginOps(mlir::MLIRContext &context) : builder(&context),
    typeTranslator(context), bbTranslator(new Detail::BlockFromGimpleTranslatorImpl(context))
{}

GimpleToPluginOps::~GimpleToPluginOps()
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
            fprintf(stderr, "tcc_comparison: %d not suppoted!\n", ccode);
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
            fprintf(stderr, "tcc_comparison not suppoted!\n");
            break;
    }
    // FIXME.
    return LT_EXPR;
}

static IExprCode TranslateExprCode(enum tree_code ccode)
{
    switch (ccode) {
        case PLUS_EXPR:
            return IExprCode::Plus;
        case MINUS_EXPR:
            return IExprCode::Minus;
        case MULT_EXPR:
            return IExprCode::Mult;
        case POINTER_PLUS_EXPR:
            return IExprCode::PtrPlus;
        case MIN_EXPR:
            return IExprCode::Min;
        case MAX_EXPR:
            return IExprCode::Max;
        case BIT_IOR_EXPR:
            return IExprCode::BitIOR;
        case BIT_XOR_EXPR:
            return IExprCode::BitXOR;
        case BIT_AND_EXPR:
            return IExprCode::BitAND;
        case LSHIFT_EXPR:
            return IExprCode::Lshift;
        case RSHIFT_EXPR:
            return IExprCode::Rshift;
        case NOP_EXPR:
            return IExprCode::Nop;
        default:
            break;
    }
    return IExprCode::UNDEF;
}

static enum tree_code TranslateExprCodeToTreeCode(IExprCode ccode)
{
    switch (ccode) {
        case IExprCode::Plus:
            return PLUS_EXPR;
        case IExprCode::Minus:
            return MINUS_EXPR;
        case IExprCode::Mult:
            return MULT_EXPR;
        case IExprCode::PtrPlus:
            return POINTER_PLUS_EXPR;
        case IExprCode::Min:
            return MIN_EXPR;
        case IExprCode::Max:
            return MAX_EXPR;
        case IExprCode::BitIOR:
            return BIT_IOR_EXPR;
        case IExprCode::BitXOR:
            return BIT_XOR_EXPR;
        case IExprCode::BitAND:
            return BIT_AND_EXPR;
        case IExprCode::Lshift:
            return LSHIFT_EXPR;
        case IExprCode::Rshift:
            return RSHIFT_EXPR;
        case IExprCode::Nop:
            return NOP_EXPR;
        default:
            break;
    }
    // FIXME.
    return NOP_EXPR;
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

string GimpleToPluginOps::DeclSourceFile(uint64_t gccDataAddr)
{
    tree decl = (tree)gccDataAddr;
    string sourceFile = DECL_SOURCE_FILE(decl);
    return sourceFile;
}

string GimpleToPluginOps::GetVariableName(uint64_t gccDataAddr)
{
    tree decl = (tree)gccDataAddr;
    string pointer = DECL_NAME(decl) != NULL_TREE ? IDENTIFIER_POINTER(DECL_NAME(decl)) : "<unamed>";
    return pointer;
}

string GimpleToPluginOps::GetFuncName(uint64_t gccDataAddr)
{
    string funcName = function_name((function *)gccDataAddr);
    return funcName;
}

int GimpleToPluginOps::DeclSourceLine(uint64_t gccDataAddr)
{
    tree decl = (tree)gccDataAddr;
    int line = DECL_SOURCE_LINE(decl);
    return line;
}

int GimpleToPluginOps::DeclSourceColumn(uint64_t gccDataAddr)
{
    tree decl = (tree)gccDataAddr;
    int column = DECL_SOURCE_COLUMN(decl);
    return column;
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

void GimpleToPluginOps::DeleteBlock(uint64_t funcAddr, uint64_t bbAddr)
{
    basic_block address = reinterpret_cast<basic_block>(bbAddr);
    function *fn = reinterpret_cast<function *>(funcAddr);
    push_cfun(fn);
    delete_basic_block(address);
    pop_cfun();
}

void GimpleToPluginOps::SetImmediateDominator(uint64_t dir, uint64_t bbAddr,
                                              uint64_t domiAddr)
{
    basic_block bb = reinterpret_cast<basic_block>(bbAddr);
    basic_block dominated = reinterpret_cast<basic_block>(domiAddr);
    if (dir == 1) {
        set_immediate_dominator(CDI_DOMINATORS, bb, dominated);
    } else if (dir == 2) {
        set_immediate_dominator(CDI_POST_DOMINATORS, bb, dominated);
    } else {
        abort();
    }
}

uint64_t GimpleToPluginOps::GetImmediateDominator(uint64_t dir, uint64_t bbAddr)
{
    basic_block bb = reinterpret_cast<basic_block>(bbAddr);
    if (dir == 1) {
        basic_block res = get_immediate_dominator(CDI_DOMINATORS, bb);
        return reinterpret_cast<uint64_t>(res);
    } else if (dir == 2) {
        basic_block res = get_immediate_dominator(CDI_POST_DOMINATORS, bb);
        return reinterpret_cast<uint64_t>(res);
    }

    abort();
}

uint64_t GimpleToPluginOps::RecomputeDominator(uint64_t dir, uint64_t bbAddr)
{
    basic_block bb = reinterpret_cast<basic_block>(bbAddr);
    if (dir == 1) {
        basic_block res = recompute_dominator(CDI_DOMINATORS, bb);
        return reinterpret_cast<uint64_t>(res);
    } else if (dir == 2) {
        basic_block res = recompute_dominator(CDI_POST_DOMINATORS, bb);
        return reinterpret_cast<uint64_t>(res);
    }

    abort();
}

// CGnode =========

CGnodeOp GimpleToPluginOps::BuildCGnodeOp(uint64_t id)
{
    cgraph_node *node;
    node = reinterpret_cast<cgraph_node *>(id);
    mlir::StringRef symbolName(node->name());
    bool definition = false;
    if (node->definition)
        definition = true;
    uint32_t order = node->order;
    auto location = builder.getUnknownLoc();
    CGnodeOp retOp = builder.create<CGnodeOp>(location, id, symbolName, definition, order);
    return retOp;
}

vector<uint64_t> GimpleToPluginOps::GetCGnodeIDs()
{
    cgraph_node *node = NULL;
    vector<uint64_t> cgnodeIDs;
    FOR_EACH_FUNCTION (node) {
        int64_t id = reinterpret_cast<int64_t>(reinterpret_cast<void*>(node));
        cgnodeIDs.push_back(id);
    }
    return cgnodeIDs;
}

CGnodeOp GimpleToPluginOps::GetCGnodeOpById(uint64_t id)
{
    CGnodeOp cgOp = BuildCGnodeOp(id);
    return cgOp;
}

bool GimpleToPluginOps::IsRealSymbolOfCGnode(uint64_t id)
{
    cgraph_node *node;
    node = reinterpret_cast<cgraph_node *>(id);
    return node->real_symbol_p();
}

//=================

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

vector<uint64_t> GimpleToPluginOps::GetFunctionIDs()
{
    cgraph_node *node = NULL;
    function *fn = NULL;
    vector<uint64_t> functions;
    FOR_EACH_FUNCTION (node) {
        if (!node->real_symbol_p ())
            continue;
        if(!node->definition) continue;
        fn = DECL_STRUCT_FUNCTION(node->decl);
        if (fn == NULL)
            continue;
        int64_t id = reinterpret_cast<int64_t>(reinterpret_cast<void*>(fn));
        functions.push_back(id);
    }
    return functions;
}

FunctionOp GimpleToPluginOps::GetFunctionById(uint64_t id)
{
    FunctionOp irFunc = BuildFunctionOp(id);
    builder.setInsertionPointAfter(irFunc.getOperation());
    return irFunc;
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

vector<DeclBaseOp> GimpleToPluginOps::GetFuncDecls(uint64_t funcID)
{
    function *fn = reinterpret_cast<function *>(funcID);
    vector<DeclBaseOp> decls;
    if (!vec_safe_is_empty(fn->local_decls)) {
        unsigned ix = 0;
        tree var = NULL_TREE;
        FOR_EACH_LOCAL_DECL (fn, ix, var) {
            uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(var));
            if (TREE_CODE(var) != VAR_DECL || !DECL_NAME (var)) {
                continue;
            }
            bool addressable = TREE_ADDRESSABLE(var);
            bool used = TREE_USED(var);
            int32_t uid = DECL_UID(var);
            mlir::Value initial = TreeToValue((uint64_t)DECL_INITIAL(var));
            mlir::Value name = TreeToValue((uint64_t)DECL_NAME(var));
            llvm::Optional<uint64_t> chain = (uint64_t)DECL_CHAIN(var);
            bool readOnly = false;
            PluginTypeBase rPluginType = PluginUndefType::get(builder.getContext());
            GetTreeAttr(id, readOnly, rPluginType);
            DeclBaseOp decl = builder.create<DeclBaseOp>(
                builder.getUnknownLoc(), id, IDefineCode::Decl, readOnly, addressable, used, uid, initial, name,
                chain, rPluginType);

            decls.push_back(decl);
        }
    }
    return decls;
}

mlir::Value GimpleToPluginOps::MakeNode(IDefineCode defcode)
{   
    enum tree_code code;
    switch (defcode) {
        case IDefineCode::FieldDecl : {
            code = FIELD_DECL;
            break;
        }
        default : {
            code = FIELD_DECL;
            break;
        }
    }
    tree field = make_node(code);
    mlir::Value v = TreeToValue(reinterpret_cast<uint64_t>(reinterpret_cast<void*>(field)));
    return v;
}

vector<FieldDeclOp> GimpleToPluginOps::GetFields(uint64_t declID)
{
    vector<FieldDeclOp> fields;
    tree decl = reinterpret_cast<tree>(declID);
    tree type = TREE_TYPE(decl);
    while (POINTER_TYPE_P (type) || TREE_CODE (type) == ARRAY_TYPE)
        type = TREE_TYPE (type);

    for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field)) {
        if (TREE_CODE (field) != FIELD_DECL) {
            continue;
        }
        uint64_t treeId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(field));
        bool addressable = TREE_ADDRESSABLE(field);
        bool used = TREE_USED(field);
        int32_t uid = DECL_UID(field);
        mlir::Value initial = TreeToValue((uint64_t)DECL_INITIAL(field));
        mlir::Value name = TreeToValue((uint64_t)DECL_NAME(field));
        uint64_t chain = (uint64_t)DECL_CHAIN(field);
        mlir::Value fieldOffset = TreeToValue((uint64_t)DECL_FIELD_OFFSET(field));
        mlir::Value fieldBitOffset = TreeToValue((uint64_t)DECL_FIELD_BIT_OFFSET(field));
        bool readOnly = false;
        PluginTypeBase rPluginType = PluginUndefType::get(builder.getContext());
        GetTreeAttr(treeId, readOnly, rPluginType);
        FieldDeclOp opValue = builder.create<FieldDeclOp>(
            builder.getUnknownLoc(), treeId, IDefineCode::FieldDecl, readOnly, addressable, used, uid, initial, name,
            chain, fieldOffset, fieldBitOffset, rPluginType);
        fields.push_back(opValue);
    }
    return fields;
}

PluginIR::PluginTypeBase GimpleToPluginOps::GetDeclType(uint64_t declID)
{
    tree decl = reinterpret_cast<tree>(declID);
    tree type = TREE_TYPE(decl);
    PluginIR::PluginTypeBase retType = typeTranslator.translateType(reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(type)));
    return retType;
}

DeclBaseOp GimpleToPluginOps::BuildDecl(IDefineCode code, string name, PluginTypeBase type)
{
    tree newtype = make_node(RECORD_TYPE);
    tree t = build_decl(UNKNOWN_LOCATION, TYPE_DECL, get_identifier(name.c_str()), newtype);
    TYPE_NAME(newtype) = t;

    uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(t));

    bool addressable = TREE_ADDRESSABLE(t);
    bool used = TREE_USED(t);
    int32_t uid = DECL_UID(t);
    mlir::Value initial = TreeToValue((uint64_t)DECL_INITIAL(t));
    mlir::Value tname = TreeToValue((uint64_t)DECL_NAME(t));
    llvm::Optional<uint64_t> chain = (uint64_t)DECL_CHAIN(t);
    bool readOnly = false;
    PluginTypeBase rPluginType = PluginUndefType::get(builder.getContext());
    GetTreeAttr(id, readOnly, rPluginType);
    DeclBaseOp decl = builder.create<DeclBaseOp>(
        builder.getUnknownLoc(), id, code, readOnly, addressable, used, uid, initial, tname,
        chain, rPluginType);
    return decl;
}

void GimpleToPluginOps::SetDeclName(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    DECL_NAME (newfield) = DECL_NAME (field);
}

void GimpleToPluginOps::SetDeclType(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    TREE_TYPE (newfield) = TREE_TYPE (field);
}

void GimpleToPluginOps::SetSourceLocation(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    DECL_SOURCE_LOCATION (newfield) = DECL_SOURCE_LOCATION (field);
}

void GimpleToPluginOps::SetDeclAlign(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    SET_DECL_ALIGN (newfield, DECL_ALIGN (field));
}

void GimpleToPluginOps::SetUserAlign(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    DECL_USER_ALIGN (newfield) = DECL_USER_ALIGN (field);
}

void GimpleToPluginOps::SetTypeFields(uint64_t declId, uint64_t fieldId)
{
    tree decl = reinterpret_cast<tree>(declId);
    tree field = reinterpret_cast<tree>(fieldId);
    TYPE_FIELDS (TREE_TYPE(decl)) = field; 
}

void GimpleToPluginOps::LayoutType(uint64_t declId)
{
    tree decl = reinterpret_cast<tree>(declId);
    layout_type (TREE_TYPE(decl));
}

void GimpleToPluginOps::LayoutDecl(uint64_t declId)
{
    tree decl = reinterpret_cast<tree>(declId);
    layout_decl (decl, 0);
    // debug_tree(decl);
    // debug_tree(TREE_TYPE(decl));
}

void GimpleToPluginOps::SetDeclChain(uint64_t newfieldId, uint64_t fieldId)
{
     tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    DECL_CHAIN (newfield) = field;     
}

unsigned GimpleToPluginOps::GetDeclTypeSize(uint64_t declId)
{
    tree decl = reinterpret_cast<tree>(declId);
    return tree_to_uhwi (TYPE_SIZE (TREE_TYPE (decl)));
}

void GimpleToPluginOps::SetAddressable(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    TREE_ADDRESSABLE (newfield) = TREE_ADDRESSABLE (field);
}

void GimpleToPluginOps::SetNonAddressablep(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    DECL_NONADDRESSABLE_P (newfield) = !TREE_ADDRESSABLE (field);
}

void GimpleToPluginOps::SetVolatile(uint64_t newfieldId, uint64_t fieldId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree field = reinterpret_cast<tree>(fieldId);
    TREE_THIS_VOLATILE (newfield) = TREE_THIS_VOLATILE (field);
}

void GimpleToPluginOps::SetDeclContext(uint64_t newfieldId, uint64_t declId)
{
    tree newfield = reinterpret_cast<tree>(newfieldId);
    tree decl = reinterpret_cast<tree>(declId);
    DECL_CONTEXT (newfield) = TREE_TYPE(decl);
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

void GimpleToPluginOps::AddBlockToLoop(uint64_t blockID, uint64_t loopID)
{
    basic_block bb = reinterpret_cast<basic_block>(reinterpret_cast<void *>(blockID));
    class loop *loop = reinterpret_cast<class loop *>(reinterpret_cast<void *>(loopID));
    add_bb_to_loop(bb, loop);
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

void GimpleToPluginOps::SetHeader(uint64_t loopID, uint64_t blockID)
{
    class loop *loop = reinterpret_cast<class loop *>(reinterpret_cast<void *>(loopID));
    basic_block bb = reinterpret_cast<basic_block>(reinterpret_cast<void *>(blockID));
    loop->header = bb;
}

void GimpleToPluginOps::SetLatch(uint64_t loopID, uint64_t blockID)
{
    class loop *loop = reinterpret_cast<class loop *>(reinterpret_cast<void *>(loopID));
    basic_block bb = reinterpret_cast<basic_block>(reinterpret_cast<void *>(blockID));
    loop->latch = bb;
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

void GimpleToPluginOps::RedirectFallthroughTarget(uint64_t src, uint64_t dest)
{
    basic_block srcbb = reinterpret_cast<basic_block>(reinterpret_cast<void*>(src));
    basic_block destbb = reinterpret_cast<basic_block>(reinterpret_cast<void*>(dest));
    assert(single_succ_p(srcbb));
    redirect_edge_and_branch (single_succ_edge(srcbb), destbb);
}

void GimpleToPluginOps::RemoveEdge(uint64_t src, uint64_t dest)
{
    basic_block srcbb = reinterpret_cast<basic_block>(reinterpret_cast<void*>(src));
    basic_block destbb = reinterpret_cast<basic_block>(reinterpret_cast<void*>(dest));
    edge e = find_edge(srcbb, destbb);
    assert(e);
    remove_edge(e);
}

bool GimpleToPluginOps::IsLtoOptimize()
{
    return in_lto_p;
}

bool GimpleToPluginOps::IsWholeProgram()
{
    return flag_whole_program;
}
static function *fn = nullptr;
FunctionOp GimpleToPluginOps::BuildFunctionOp(uint64_t functionId)
{
    fn = reinterpret_cast<function*>(functionId);
    mlir::StringRef funcName(function_name(fn));
    bool declaredInline = false;
    if (DECL_DECLARED_INLINE_P(fn->decl))
        declaredInline = true;
    auto location = builder.getUnknownLoc();
    bool validType = false;
    tree returnType = TREE_TYPE(fn->decl);
    FunctionOp retOp;
    if (TREE_CODE(returnType) != FUNCTION_TYPE) {
        retOp = builder.create<FunctionOp>(location, functionId,
                                        funcName, declaredInline, validType);
    } else {
        validType = true;
        PluginTypeBase rPluginType = typeTranslator.translateType((intptr_t)returnType);
        auto Ty = rPluginType.dyn_cast<PluginFunctionType>();
        retOp = builder.create<FunctionOp>(location, functionId,
                                        funcName, declaredInline, Ty, validType);
    }
    auto& fr = retOp.bodyRegion();
    if (fn->cfg == nullptr) return retOp;
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
            assert(EDGE_COUNT(stmt->bb->succs) == 2);
            Block* trueBlock = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 0)->dest];
            Block* falseBlock = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 1)->dest];
            CondOp condOp = BuildCondOp(id, (uint64_t)stmt->bb,
                                        trueBlock, falseBlock,
                                        (uint64_t)EDGE_SUCC(stmt->bb, 0)->dest,
                                        (uint64_t)EDGE_SUCC(stmt->bb, 1)->dest);
            ret = condOp.getOperation();
            break;
        }
        case GIMPLE_DEBUG: {
            DebugOp debugOp = builder.create<DebugOp>(
                    builder.getUnknownLoc(), id);
            ret = debugOp.getOperation();
            break;
        }
        case GIMPLE_ASM: {
            AsmOp asmOp = BuildAsmOp(id);
            ret = asmOp.getOperation();
            break;
        }
        case GIMPLE_SWITCH: {
            SwitchOp switchOp = BuildSwitchOp(id);
            ret = switchOp.getOperation();
            break;
        }
        case GIMPLE_LABEL: {
            LabelOp labelOp = BuildLabelOp(id);
            ret = labelOp.getOperation();
            break;
        }
        case GIMPLE_GOTO: {
            Block* success = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 0)->dest];
            GotoOp gotoOp = BuildGotoOp(id, (uint64_t)stmt->bb, success, (uint64_t)EDGE_SUCC(stmt->bb, 0)->dest);
            ret = gotoOp.getOperation();
            break;
        }
        case GIMPLE_TRANSACTION: {
            TransactionOp tranOp = BuildTransactionOp(id);
            ret = tranOp.getOperation();
            break;
        }
        case GIMPLE_RESX: {
            ResxOp resxOp = BuildResxOp(id);
            ret = resxOp.getOperation();
            break;
        }
        case GIMPLE_TRY: {
            fprintf(stderr, "try stmt \n");
            TryOp tryOp = BuildTryOp(id);
            ret = tryOp.getOperation();
            break;
        }
        case GIMPLE_CATCH: {
            fprintf(stderr, "catch stmt \n");
            CatchOp catchOp = BuildCatchOp(id);
            ret = catchOp.getOperation();
            break;
        }
        case GIMPLE_BIND: {
            fprintf(stderr, "bind stmt \n");
            BindOp bindOp = BuildBindOp(id);
            ret = bindOp.getOperation();
            break;
        }
        case GIMPLE_EH_MUST_NOT_THROW: {
            EHMntOp ehMntOp = BuildEHMntOp(id);
            ret = ehMntOp.getOperation();
            break;
        }
        case GIMPLE_EH_DISPATCH: {
            EHDispatchOp dispatchOp = BuildEHDispatchOp(id);
            ret = dispatchOp.getOperation();
            break;
        }
        case GIMPLE_NOP : {
            NopOp nopOp = BuildNopOp(id);
            ret = nopOp.getOperation();
            break;
        }
        case GIMPLE_EH_ELSE : {
            EHElseOp ehElseOp = BuildEHElseOp(id);
            ret = ehElseOp.getOperation();
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
        tree argTree = gimple_phi_arg_def(stmt, i);
        if (argTree == NULL_TREE) continue;
        uint64_t argId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(argTree));
        Value arg = TreeToValue(argId);
        ops.push_back(arg);
    }
    PluginTypeBase rPluginType = nullptr;
    tree retTree = gimple_phi_result(stmt);
    if (retTree != NULL_TREE) {
        tree returnType = TREE_TYPE(retTree);
        rPluginType = typeTranslator.translateType((intptr_t)returnType);
    }
    uint32_t capacity = gimple_phi_capacity(stmt);
    uint32_t nArgs = gimple_phi_num_args(stmt);
    PhiOp ret = builder.create<PhiOp>(builder.getUnknownLoc(),
                                      gphiId, capacity, nArgs, ops, rPluginType);
    return ret;
}

uint64_t GimpleToPluginOps::CreateGphiNode(uint64_t argId, uint64_t blockId)
{
    tree arg = NULL_TREE;
    if (argId != 0) {
        arg = reinterpret_cast<tree>(argId);
    }
    basic_block bb = reinterpret_cast<basic_block>(blockId);
    gphi *ret = create_phi_node(arg, bb);
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(ret));
}

CallOp GimpleToPluginOps::BuildCallOp(uint64_t gcallId)
{
    gcall *stmt = reinterpret_cast<gcall*>(gcallId);
    llvm::SmallVector<Value, 4> ops;
    ops.reserve(gimple_call_num_args(stmt));
    for (unsigned i = 0; i < gimple_call_num_args(stmt); i++) {
        tree callArg = gimple_call_arg(stmt, i);
        uint64_t argId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(callArg));
        Value arg = TreeToValue(argId);
        ops.push_back(arg);
    }
    // tree returnType = gimple_call_return_type(stmt);
    // PluginTypeBase rPluginType = typeTranslator.translateType((intptr_t)returnType);
    PluginTypeBase rPluginType = nullptr;
    if (gimple_call_fntype (stmt) != NULL_TREE || gimple_call_lhs (stmt) != NULL_TREE) {
        tree returnType = gimple_call_return_type(stmt);
        rPluginType = typeTranslator.translateType((intptr_t)returnType);
    }
    tree fndecl = gimple_call_fndecl(stmt);
    CallOp ret;
    if (fndecl == NULL_TREE || DECL_NAME(fndecl) == NULL_TREE) {
        ret = builder.create<CallOp>(builder.getUnknownLoc(),
                                     gcallId, (uint64_t)stmt->bb, ops, rPluginType);
    } else {
        StringRef callName(IDENTIFIER_POINTER(DECL_NAME(fndecl)));
        ret = builder.create<CallOp>(builder.getUnknownLoc(),
                                     gcallId, (uint64_t)stmt->bb, callName, ops, rPluginType);
    }    return ret;
}

bool GimpleToPluginOps::SetGimpleCallLHS(uint64_t callId, uint64_t lhsId)
{
    gcall *stmt = reinterpret_cast<gcall*>(callId);
    tree lhs = reinterpret_cast<tree>(lhsId);
    gimple_call_set_lhs (stmt, lhs);
    return true;
}

uint64_t GimpleToPluginOps::CreateGcallVec(uint64_t blockId, uint64_t funcId,
                                           vector<uint64_t> &argIds)
{
    tree fn;
    if (funcId) {
        fn = reinterpret_cast<tree>(funcId);
    } else {
        // FIXME.
        fn = builtin_decl_explicit (BUILT_IN_CTZLL);
    }
    auto_vec<tree> vargs (argIds.size());
    for (auto id : argIds) {
        tree arg = reinterpret_cast<tree>(id);
        vargs.quick_push (arg);
    }
    gcall *ret = gimple_build_call_vec (fn, vargs);
    basic_block bb = reinterpret_cast<basic_block>(blockId);
    if (bb != nullptr) {
        gimple_stmt_iterator si = gsi_last_bb (bb);
        gsi_insert_after (&si, ret, GSI_NEW_STMT);
    }
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(ret));
}

uint32_t GimpleToPluginOps::AddPhiArg(uint64_t phiId, uint64_t argId, uint64_t predId, uint64_t succId)
{
    gphi *phi = reinterpret_cast<gphi*>(phiId);
    tree arg = reinterpret_cast<tree>(argId);
    basic_block pred = reinterpret_cast<basic_block>(predId);
    basic_block succ = reinterpret_cast<basic_block>(succId);
    edge e = find_edge(pred, succ);
    location_t loc;
    if (virtual_operand_p (arg))
    loc = UNKNOWN_LOCATION;
    else
    loc = gimple_location (SSA_NAME_DEF_STMT (arg));
    add_phi_arg (phi, arg, e, loc);
    return gimple_phi_num_args(phi);
}

uint64_t GimpleToPluginOps::CreateGassign(uint64_t blockId, IExprCode iCode,
                                          vector<uint64_t> &argIds)
{
    vector<tree> vargs;
    for (auto id : argIds) {
        tree arg = reinterpret_cast<tree>(id);
        vargs.push_back (arg);
    }
    gassign *ret;
    if (vargs.size() == 2) {
        if (iCode == IExprCode::UNDEF) {
            ret = gimple_build_assign(vargs[0], vargs[1]);
        } else {
            ret = gimple_build_assign(vargs[0], TranslateExprCodeToTreeCode(iCode), vargs[1]);
        }
    } else if (vargs.size() == 3) {
        ret = gimple_build_assign(vargs[0], TranslateExprCodeToTreeCode(iCode), vargs[1], vargs[2]);
    } else if (vargs.size() == 4) {
        ret = gimple_build_assign(vargs[0], TranslateExprCodeToTreeCode(iCode), vargs[1], vargs[2], vargs[3]);
    } else {
        fprintf(stderr, "ERROR size: %ld.\n", vargs.size());
    }
    basic_block bb = reinterpret_cast<basic_block>(blockId);
    if (bb != nullptr) {
        gimple_stmt_iterator si = gsi_last_bb (bb);
        gsi_insert_after (&si, ret, GSI_NEW_STMT);
    }
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(ret));
}

uint64_t GimpleToPluginOps::CreateGcond(uint64_t blockId, IComparisonCode iCode,
    uint64_t lhsId, uint64_t rhsId, uint64_t tbaddr, uint64_t fbaddr)
{
    tree lhs = reinterpret_cast<tree>(lhsId);
    tree rhs = reinterpret_cast<tree>(rhsId);
    gcond *ret = gimple_build_cond (TranslateCmpCodeToTreeCode(iCode),
                                    lhs, rhs, NULL_TREE, NULL_TREE);
    basic_block bb = reinterpret_cast<basic_block>(blockId);
    if (bb != nullptr) {
        gimple_stmt_iterator si = gsi_last_bb (bb);
        gsi_insert_after (&si, ret, GSI_NEW_STMT);
    }
    basic_block tb = reinterpret_cast<basic_block>(tbaddr);
    basic_block fb = reinterpret_cast<basic_block>(fbaddr);
    assert(make_edge(bb, tb, EDGE_TRUE_VALUE));
    assert(make_edge(bb, fb, EDGE_FALSE_VALUE));
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(ret));
}

void GimpleToPluginOps::CreateFallthroughOp(uint64_t addr, uint64_t destaddr)
{
    basic_block src = reinterpret_cast<basic_block>(addr);
    basic_block dest = reinterpret_cast<basic_block>(destaddr);
    assert(make_single_succ_edge(src, dest, EDGE_FALLTHRU));
}

CondOp GimpleToPluginOps::BuildCondOp(uint64_t gcondId, uint64_t address,
    Block* b1, Block* b2, uint64_t tbaddr, uint64_t fbaddr)
{
    gcond *stmt = reinterpret_cast<gcond*>(gcondId);
    tree lhsPtr = gimple_cond_lhs(stmt);
    uint64_t lhsId = reinterpret_cast<uint64_t>(
        reinterpret_cast<void*>(lhsPtr));
    Value LHS = TreeToValue(lhsId);
    tree rhsPtr = gimple_cond_rhs(stmt);
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
            reinterpret_cast<void*>(gimple_assign_lhs(stmt)));
    ops.push_back(TreeToValue(lhsId));
    uint64_t rhs1Id = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_assign_rhs1(stmt)));
    ops.push_back(TreeToValue(rhs1Id));
    if (gimple_assign_rhs2(stmt) != NULL_TREE) {
        uint64_t rhs2Id = reinterpret_cast<uint64_t>(
                reinterpret_cast<void*>(gimple_assign_rhs2(stmt)));
        ops.push_back(TreeToValue(rhs2Id));
    }
    if (gimple_assign_rhs3(stmt) != NULL_TREE) {
        uint64_t rhs3Id = reinterpret_cast<uint64_t>(
                reinterpret_cast<void*>(gimple_assign_rhs3(stmt)));
        ops.push_back(TreeToValue(rhs3Id));
    }
    IExprCode iCode = TranslateExprCode(gimple_assign_rhs_code(stmt));
    tree returnType = TREE_TYPE(gimple_assign_lhs(stmt));
    PluginTypeBase rPluginType = typeTranslator.translateType((intptr_t)returnType);
    AssignOp ret = builder.create<AssignOp>(
            builder.getUnknownLoc(), gassignId, iCode, ops, rPluginType);
    return ret;
}

NopOp GimpleToPluginOps::BuildNopOp(uint64_t gnopId)
{
    gimple *stmt = reinterpret_cast<gimple*>(gnopId);
    NopOp ret = builder.create<NopOp>(builder.getUnknownLoc(), gnopId);
    return ret;
}

EHElseOp GimpleToPluginOps::BuildEHElseOp(uint64_t geh_elseId)
{
    geh_else *stmt = reinterpret_cast<geh_else*>(geh_elseId);

    llvm::SmallVector<uint64_t, 4> nbodyaddrs, ebodyaddrs;
    gimple_seq body = gimple_eh_else_n_body(stmt);
    gimple_stmt_iterator gsi;
     for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        nbodyaddrs.push_back(stmtId);
    }
    body = gimple_eh_else_e_body(stmt);
    for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        ebodyaddrs.push_back(stmtId);
    }
    EHElseOp ret = builder.create<EHElseOp>(builder.getUnknownLoc(), geh_elseId, nbodyaddrs, ebodyaddrs);
    return ret;
}

AsmOp GimpleToPluginOps::BuildAsmOp(uint64_t gasmId)
{
    gasm *stmt = reinterpret_cast<gasm*>(gasmId);
    llvm::SmallVector<Value, 4> ops;
    llvm::StringRef statement(gimple_asm_string(stmt));
    uint32_t nInputs = gimple_asm_ninputs(stmt);
    uint32_t nOuputs = gimple_asm_noutputs(stmt);
    uint32_t nClobbers = gimple_asm_nclobbers(stmt);

    for (size_t i = 0; i < nInputs; i++) {
        uint64_t input = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_asm_input_op(stmt, i)));
        Value tett = TreeToValue(input);
        ops.push_back(tett);
    }

    for (size_t i = 0; i < nOuputs; i++) {
        uint64_t input = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_asm_output_op(stmt, i)));
        ops.push_back(TreeToValue(input));
    }

    for (size_t i = 0; i < nClobbers; i++) {
        uint64_t input = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_asm_clobber_op(stmt, i)));
        ops.push_back(TreeToValue(input));
    }

    AsmOp ret = builder.create<AsmOp>(
            builder.getUnknownLoc(), gasmId, statement, nInputs, nOuputs, nClobbers, ops);
    return ret;
}

SwitchOp GimpleToPluginOps::BuildSwitchOp(uint64_t gswitchId)
{
    gswitch *stmt = reinterpret_cast<gswitch*>(gswitchId);
    llvm::SmallVector<Value, 4> ops;

    uint64_t sIndex = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_switch_index(stmt)));
    Value index = TreeToValue(sIndex);

    uint64_t sDefault = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_switch_default_label(stmt)));
    Value defaultLabel = TreeToValue(sDefault);

    unsigned nLabels = gimple_switch_num_labels(stmt);
    for (size_t i = 1; i < nLabels; i++) {
        uint64_t input = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_switch_label(stmt, i)));
        ops.push_back(TreeToValue(input));
    }

    Block *defaultDest = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 0)->dest];
    llvm::SmallVector<Block*, 4> caseDest;
    llvm::SmallVector<uint64_t, 4> caseaddr;

    push_cfun(fn);
    for (size_t i = 1; i < nLabels; i++) {
        basic_block label_bb = gimple_switch_label_bb (cfun, stmt, i);
        Block *temp = bbTranslator->blockMaps[label_bb];
        caseaddr.push_back((uint64_t)label_bb);
        caseDest.push_back(temp);
    }
    pop_cfun();
    SwitchOp ret = builder.create<SwitchOp>(
            builder.getUnknownLoc(), gswitchId, index, (uint64_t)(stmt->bb), defaultLabel, ops, 
                defaultDest, (uint64_t)(EDGE_SUCC(stmt->bb, 0)->dest), caseDest, caseaddr);
    return ret;
}

LabelOp GimpleToPluginOps::BuildLabelOp(uint64_t glabelId)
{
    glabel *stmt = reinterpret_cast<glabel*>(glabelId);
    llvm::SmallVector<Value, 4> ops;

    uint64_t labeladdr = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_label_label(stmt)));
    Value label = TreeToValue(labeladdr);

    LabelOp ret = builder.create<LabelOp>(
            builder.getUnknownLoc(), glabelId, label);
    return ret;
}

EHMntOp GimpleToPluginOps::BuildEHMntOp(uint64_t gehmntId)
{
    geh_mnt *stmt = reinterpret_cast<geh_mnt*>(gehmntId);
    llvm::SmallVector<Value, 4> ops;

    uint64_t fndecladdr = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_eh_must_not_throw_fndecl(stmt)));
    Value fndecl = TreeToValue(fndecladdr);
    fprintf(stderr, "build --------------------------------------\n");
    EHMntOp ret = builder.create<EHMntOp>(
            builder.getUnknownLoc(), gehmntId, fndecl);
    return ret;
}

GotoOp GimpleToPluginOps::BuildGotoOp(uint64_t ggotoId, uint64_t address, Block* success, uint64_t successaddr)
{
    ggoto *stmt = reinterpret_cast<ggoto*>(ggotoId);
    llvm::SmallVector<Value, 4> ops;

    uint64_t destaddr = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_goto_dest(stmt)));
    Value dest = TreeToValue(destaddr);


    GotoOp ret = builder.create<GotoOp>(
            builder.getUnknownLoc(), ggotoId, address, dest, success, successaddr);
    return ret;
}

TransactionOp GimpleToPluginOps::BuildTransactionOp(uint64_t ggtransaction)
{
    gtransaction *stmt = reinterpret_cast<gtransaction*>(ggtransaction);
    llvm::SmallVector<uint64_t, 4> stmtaddr;
    gimple_seq body = gimple_transaction_body(stmt);
    gimple_stmt_iterator gsi;
     for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        stmtaddr.push_back(stmtId);
    }
    tree label = gimple_transaction_label_norm (stmt);
    tree uinst = gimple_transaction_label_uninst (stmt);
    tree over = gimple_transaction_label_over (stmt);
    Value labelNorm = TreeToValue(reinterpret_cast<uint64_t>(reinterpret_cast<void*>(label)));
    Value labelUninst = TreeToValue(reinterpret_cast<uint64_t>(reinterpret_cast<void*>(uinst)));
    Value labelOver = TreeToValue(reinterpret_cast<uint64_t>(reinterpret_cast<void*>(over)));
    Block *fallthrough, *abort;
    uint64_t fallthroughaddr, abortaddr;
    assert(EDGE_COUNT(stmt->bb->succs) == 2);
    fallthrough = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 0)->dest];
    fallthroughaddr = (uint64_t)(EDGE_SUCC(stmt->bb, 0)->dest);
    abort = bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, 1)->dest];
    abortaddr = (uint64_t)(EDGE_SUCC(stmt->bb, 1)->dest);

    TransactionOp ret = builder.create<TransactionOp>(
            builder.getUnknownLoc(), ggtransaction, (uint64_t)stmt->bb, stmtaddr, labelNorm, labelUninst, labelOver,
            fallthrough, fallthroughaddr, abort, abortaddr);
    return ret;
}

EHDispatchOp GimpleToPluginOps::BuildEHDispatchOp(uint64_t geh_dispatchId)
{
    geh_dispatch *stmt = reinterpret_cast<geh_dispatch*>(geh_dispatchId);
    uint64_t region = gimple_eh_dispatch_region(stmt);
    llvm::SmallVector<Block*, 4> ehHandlers;
    llvm::SmallVector<uint64_t, 4> ehHandlersaddr;

    for (unsigned int i = 0; i < EDGE_COUNT (stmt->bb->succs); i++) {
        ehHandlers.push_back(bbTranslator->blockMaps[EDGE_SUCC(stmt->bb, i)->dest]);
        ehHandlersaddr.push_back((uint64_t)EDGE_SUCC(stmt->bb, i)->dest);
    }

    EHDispatchOp ret = builder.create<EHDispatchOp>(
            builder.getUnknownLoc(), geh_dispatchId, (uint64_t)stmt->bb, region, ehHandlers, ehHandlersaddr);
    return ret;
}

ResxOp GimpleToPluginOps::BuildResxOp(uint64_t ggresx)
{
    gresx *stmt = reinterpret_cast<gresx*>(ggresx);
    int region = gimple_resx_region(stmt);
    ResxOp ret = builder.create<ResxOp>(builder.getUnknownLoc(), ggresx, (uint64_t)stmt->bb, region);
    return ret;
}

BindOp GimpleToPluginOps::BuildBindOp(uint64_t gbindId)
{
    gbind *stmt = reinterpret_cast<gbind*>(gbindId);
    uint64_t varsaddr = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_bind_vars(stmt)));
    Value vars = TreeToValue(varsaddr);

    llvm::SmallVector<uint64_t, 4> bodyaddrs;
    gimple_seq body = gimple_bind_body(stmt);
    gimple_stmt_iterator gsi;
     for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        bodyaddrs.push_back(stmtId);
    }
    Value block = TreeToValue(reinterpret_cast<uint64_t>(reinterpret_cast<void*>(gimple_bind_block(stmt))));
    BindOp ret = builder.create<BindOp>(builder.getUnknownLoc(), gbindId, vars, bodyaddrs, block);
    return ret;
}

TryOp GimpleToPluginOps::BuildTryOp(uint64_t gtryId)
{
    gtry *stmt = reinterpret_cast<gtry*>(gtryId);

    llvm::SmallVector<uint64_t, 4> evaladdrs, cleanupaddrs;
    gimple_seq body = gimple_try_eval(stmt);
    gimple_stmt_iterator gsi;
     for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        evaladdrs.push_back(stmtId);
    }

    body = gimple_try_cleanup(stmt);
     for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
                            
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        CatchOp catchOp = BuildCatchOp(stmtId);
        cleanupaddrs.push_back(stmtId);
    }
    uint64_t kind = gimple_try_kind(stmt);
    TryOp ret = builder.create<TryOp>(builder.getUnknownLoc(), gtryId, evaladdrs, cleanupaddrs, kind);
    return ret;
}

CatchOp GimpleToPluginOps::BuildCatchOp(uint64_t gcatchId)
{
    gcatch *stmt = reinterpret_cast<gcatch*>(gcatchId);
    uint64_t typesaddr = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(gimple_catch_types(stmt)));
    Value types = TreeToValue(typesaddr);
    llvm::SmallVector<uint64_t, 4> handleraddrs;
    gimple_seq body = gimple_catch_handler(stmt);
    gimple_stmt_iterator gsi;
     for (gsi = gsi_start (body); !gsi_end_p (gsi); gsi_next (&gsi))
    {
        gimple *stmtbody = gsi_stmt (gsi);
        uint64_t stmtId = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(stmtbody));
        handleraddrs.push_back(stmtId);
    }

    CatchOp ret = builder.create<CatchOp>(builder.getUnknownLoc(), gcatchId, types, handleraddrs);
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

Value GimpleToPluginOps::BuildIntCst(mlir::Type type, int64_t init)
{
    PluginTypeBase pluginType = type.dyn_cast<PluginTypeBase>();
    uintptr_t typeId = pluginTypeTranslator.translateType(pluginType);
    tree treeType = reinterpret_cast<tree>(typeId);
    tree ret = build_int_cst(treeType, init);
    uint64_t retId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(ret));
    return TreeToValue(retId);
}

void GimpleToPluginOps::GetTreeAttr(uint64_t treeId, bool &readOnly, PluginTypeBase &rPluginType)
{
    tree t = reinterpret_cast<tree>(treeId);
    tree treeType = TREE_TYPE(t);
    if (treeType == NULL_TREE) return;
    readOnly = TYPE_READONLY(treeType);
    uintptr_t typeId = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(treeType));
    rPluginType = typeTranslator.translateType(typeId);
}

Value GimpleToPluginOps::TreeToValue(uint64_t treeId)
{
    tree t = reinterpret_cast<tree>(treeId);
    bool readOnly = false;
    PluginTypeBase rPluginType = PluginUndefType::get(builder.getContext());
    if (t == NULL_TREE) {
        return builder.create<PlaceholderOp>(builder.getUnknownLoc(),
                    treeId, IDefineCode::UNDEF, readOnly, rPluginType);
    }
    mlir::Value opValue;
    switch (TREE_CODE(t)) {
        case INTEGER_CST : {
            mlir::Attribute initAttr;
            if (tree_fits_shwi_p(t)) {
                signed HOST_WIDE_INT sinit = tree_to_shwi(t);
                initAttr = builder.getI64IntegerAttr(sinit);
            } else if (tree_fits_uhwi_p(t)) {
                unsigned HOST_WIDE_INT uinit = tree_to_uhwi(t);
                initAttr = builder.getI64IntegerAttr(uinit);
            } else {
                abort();
            }
			GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<ConstOp>(
                    builder.getUnknownLoc(), treeId, IDefineCode::IntCST,
                    readOnly, initAttr, rPluginType);
            break;
        }
        case MEM_REF : {
            tree operand0 = TREE_OPERAND(t, 0);
            tree operand1 = TREE_OPERAND(t, 1);
            mlir::Value op0 = TreeToValue((uint64_t)operand0);
            mlir::Value op1 = TreeToValue((uint64_t)operand1);
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<MemOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::MemRef, readOnly,
                op0, op1, rPluginType);
            break;
        }
        case SSA_NAME : {
            uint64_t ssaParmDecl = 0;
            if (SSA_NAME_VAR (t) != NULL_TREE) {
                ssaParmDecl = (TREE_CODE (SSA_NAME_VAR (t)) == PARM_DECL) ? 1 : 0;
            }
            uint64_t version = SSA_NAME_VERSION(t);
            uint64_t definingId = reinterpret_cast<uint64_t>(SSA_NAME_DEF_STMT(t));
            uint64_t nameVarId = reinterpret_cast<uint64_t>(SSA_NAME_VAR(t));
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<SSAOp>(builder.getUnknownLoc(), treeId,
                                         IDefineCode::SSA, readOnly,
                                         nameVarId, ssaParmDecl, version,
                                         definingId, rPluginType);
            break;
        }
        case TREE_LIST : {
            llvm::SmallVector<Value, 4> ops;
            tree purpose = TREE_PURPOSE(t);
            tree value = TREE_VALUE(t);
            mlir::Value p = purpose == NULL_TREE ? nullptr : TreeToValue((uint64_t)purpose);
            mlir::Value v = TreeToValue((uint64_t)value);
            tree treeType = TREE_TYPE(value);
            if (treeType) {
                readOnly = TYPE_READONLY(treeType);
                uintptr_t typeId = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(treeType));
                rPluginType = typeTranslator.translateType(typeId);
            }
            bool hasPurpose = false;
            if (p) {
                hasPurpose = true;
                ops.push_back(p);
            }
            ops.push_back(v);
            opValue = builder.create<ListOp>(builder.getUnknownLoc(), treeId,
                                         IDefineCode::LIST, readOnly, hasPurpose,
                                        ops, rPluginType);
            break;
        }
        case STRING_CST : {
            llvm::StringRef str = llvm::StringRef(TREE_STRING_POINTER(t), TREE_STRING_LENGTH(t));
            tree treeType = TREE_TYPE(t);
            if (treeType != NULL_TREE) {
                readOnly = TYPE_READONLY(treeType);
                uintptr_t typeId = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(treeType));
                rPluginType = typeTranslator.translateType(typeId);
            }
            opValue = builder.create<StrOp>(builder.getUnknownLoc(), treeId,
                                            IDefineCode::StrCST, readOnly, str, rPluginType);
            break;
        }
        case IDENTIFIER_NODE : {
            llvm::StringRef str = llvm::StringRef(IDENTIFIER_POINTER(t), IDENTIFIER_LENGTH(t));
            tree treeType = TREE_TYPE(t);
            if (treeType != NULL_TREE) {
                readOnly = TYPE_READONLY(treeType);
                uintptr_t typeId = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(treeType));
                rPluginType = typeTranslator.translateType(typeId);
            }
            opValue = builder.create<StrOp>(builder.getUnknownLoc(), treeId,
                                            IDefineCode::StrCST, readOnly, str, rPluginType);
            break;
        }
        case ARRAY_REF : {
            tree operand0 = TREE_OPERAND(t, 0);
            tree operand1 = TREE_OPERAND(t, 1);
            mlir::Value op0 = TreeToValue((uint64_t)operand0);
            mlir::Value op1 = TreeToValue((uint64_t)operand1);
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<ArrayOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::ArrayRef, readOnly,
                op0, op1, rPluginType);
            break;
        }
        case VAR_DECL: {
            bool addressable = TREE_ADDRESSABLE(t);
            bool used = TREE_USED(t);
            int32_t uid = DECL_UID(t);
            // Fixme: DECL_INITIAL(t) This function causes a memory access error after repeated TreeToValue iterations.
            // postgresql-11.3 ICE
            mlir::Value initial = builder.create<PlaceholderOp>(builder.getUnknownLoc(), 0, IDefineCode::UNDEF, 0, rPluginType);
            mlir::Value name = TreeToValue((uint64_t)DECL_NAME(t));
            llvm::Optional<uint64_t> chain = (uint64_t)DECL_CHAIN(t);
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<DeclBaseOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::Decl, readOnly, addressable, used, uid, initial, name,
                chain, rPluginType);
            break;
        }
        case LABEL_DECL:
        case FUNCTION_DECL:
        case RESULT_DECL: 
        case PARM_DECL: 
        case TYPE_DECL: 
        case TRANSLATION_UNIT_DECL : {
            bool addressable = TREE_ADDRESSABLE(t);
            bool used = TREE_USED(t);
            int32_t uid = DECL_UID(t);
            mlir::Value initial = TreeToValue((uint64_t)DECL_INITIAL(t));
            mlir::Value name = TreeToValue((uint64_t)DECL_NAME(t));
            llvm::Optional<uint64_t> chain = (uint64_t)DECL_CHAIN(t);
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<DeclBaseOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::Decl, readOnly, addressable, used, uid, initial, name,
                chain, rPluginType);
            break;
        }
        case FIELD_DECL : {
            bool addressable = TREE_ADDRESSABLE(t);
            bool used = TREE_USED(t);
            int32_t uid = DECL_UID(t);
            mlir::Value initial = TreeToValue((uint64_t)DECL_INITIAL(t));
            mlir::Value name = TreeToValue((uint64_t)DECL_NAME(t));
            uint64_t chain = (uint64_t)DECL_CHAIN(t);
            mlir::Value fieldOffset = TreeToValue((uint64_t)DECL_FIELD_OFFSET(t));
            mlir::Value fieldBitOffset = TreeToValue((uint64_t)DECL_FIELD_BIT_OFFSET(t));
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<FieldDeclOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::FieldDecl, readOnly, addressable, used, uid, initial, name,
                chain, fieldOffset, fieldBitOffset, rPluginType);
            break;
        }
        case ADDR_EXPR : {
            mlir::Value operand = TreeToValue((uint64_t)TREE_OPERAND (t, 0));
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<AddressOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::AddrExp, readOnly, operand, rPluginType);
            break;
        }
        case CONSTRUCTOR : {
            int32_t len = CONSTRUCTOR_NELTS(t);
            llvm::SmallVector<Value, 4> idx, val;
            unsigned HOST_WIDE_INT cnt;
            tree index, value;
            FOR_EACH_CONSTRUCTOR_ELT (CONSTRUCTOR_ELTS (t), cnt, index, value)
            {
                mlir::Value  eleIdx= TreeToValue((uint64_t)index);
                mlir::Value  eleVal= TreeToValue((uint64_t)value);
                idx.push_back(eleIdx);
                val.push_back(eleVal);
            }
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<ConstructorOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::Constructor, readOnly, len, idx, val, rPluginType);
            break;
        }
        case TREE_VEC : {
            int32_t len = TREE_VEC_LENGTH(t);
            llvm::SmallVector<Value, 4> elements;

            for (int i = 0; i < len; i++) {
                mlir::Value ele = TreeToValue((uint64_t)TREE_VEC_ELT (t, i));
                elements.push_back(ele);
            }
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<VecOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::Vec, readOnly, len, elements, rPluginType);
            break;
        }
        case BLOCK : {
            if (BLOCK_VARS(t) == NULL_TREE) {
                GetTreeAttr(treeId, readOnly, rPluginType);
                opValue = builder.create<PlaceholderOp>(builder.getUnknownLoc(),
                        treeId, IDefineCode::UNDEF, readOnly, rPluginType);
                break;
            }
            llvm::Optional<mlir::Value> vars = TreeToValue((uint64_t)BLOCK_VARS(t));
            llvm::Optional<uint64_t> supercontext = (uint64_t)BLOCK_SUPERCONTEXT(t);
            llvm::Optional<mlir::Value> subblocks = TreeToValue((uint64_t)BLOCK_SUBBLOCKS(t));
            llvm::Optional<mlir::Value> chain = TreeToValue((uint64_t)BLOCK_CHAIN(t));
            llvm::Optional<mlir::Value> abstract_origin = TreeToValue((uint64_t)BLOCK_ABSTRACT_ORIGIN(t));
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<BlockOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::BLOCK, readOnly, vars, supercontext, subblocks,
                chain, abstract_origin, rPluginType);
            break;
        }
        case COMPONENT_REF : {
            mlir::Value component = TreeToValue((uint64_t)TREE_OPERAND (t, 0));
            mlir::Value field = TreeToValue((uint64_t)TREE_OPERAND (t, 1));
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<ComponentOp>(
                builder.getUnknownLoc(), treeId, IDefineCode::COMPONENT, readOnly, component, field, rPluginType);
            break;
        }
        default: {
            GetTreeAttr(treeId, readOnly, rPluginType);
            opValue = builder.create<PlaceholderOp>(builder.getUnknownLoc(),
                    treeId, IDefineCode::UNDEF, readOnly, rPluginType);
            break;
        }
    }
    return opValue;
}

void GimpleToPluginOps::DebugValue(uint64_t valId)
{
    tree t = reinterpret_cast<tree>(valId);
}

mlir::Value GimpleToPluginOps::BuildMemRef(PluginIR::PluginTypeBase type, uint64_t baseId, uint64_t offsetId)
{
    tree refType = (tree)pluginTypeTranslator.translateType(type);
    tree base = (tree)baseId;
    tree offset = (tree)offsetId;
    tree memRef = fold_build2(MEM_REF, refType, base, offset);
    return TreeToValue((uint64_t)memRef);
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
            fprintf(stderr, "ERROR: BuildOperation!");
        }
        if(gimple_code(stmt) == GIMPLE_COND || gimple_code(stmt) == GIMPLE_SWITCH
            || gimple_code(stmt) == GIMPLE_TRANSACTION || gimple_code(stmt) == GIMPLE_RESX ||
            gimple_code(stmt) == GIMPLE_EH_DISPATCH) {
            putTerminator = true;
        }
    }

    // EH edge and fallthrough edge, refer to gcc tree-eh source code
    if(last_stmt(bb) && gimple_code(last_stmt(bb)) == GIMPLE_CALL && EDGE_COUNT (bb->succs) == 2) {
        putTerminator = true;
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
            builder.create<FallThroughOp>(builder.getUnknownLoc(), (uint64_t)bb,
                            bbTranslator->blockMaps[EDGE_SUCC(bb, 0)->dest],
                            (uint64_t)(EDGE_SUCC(bb, 0)->dest));
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

    // create basic block
    Block* block = builder.createBlock(&rg, rg.begin());
    bbTranslator->blockMaps.insert({bb, block});
    // todo process func return type
    // todo isDeclaration

    // process succ
    for (unsigned int i = 0; i < EDGE_COUNT (bb->succs); i++) {
        if (!ProcessBasicBlock((intptr_t)(EDGE_SUCC(bb, i)->dest), rg)) {
            return false;
        }
    }
    // process each stmt
    builder.setInsertionPointToStart(block);
    if (!ProcessGimpleStmt(bbPtr, rg)) {
        return false;
    }
    return true;
}

bool GimpleToPluginOps::UpdateSSA()
{
    update_ssa(TODO_update_ssa);
    return true;
}

vector<PhiOp> GimpleToPluginOps::GetPhiOpsInsideBlock(uint64_t bb)
{
    basic_block header = reinterpret_cast<basic_block>(bb);
    vector<PhiOp> phiOps;
    gphi_iterator gsi;
    for (gsi = gsi_start_phis(header); !gsi_end_p(gsi); gsi_next(&gsi)) {
        gphi *phi = gsi.phi();
        PhiOp phiOp;
        uint64_t id = reinterpret_cast<uint64_t>(reinterpret_cast<void*>(phi));
        phiOp = BuildPhiOp(id);
        phiOps.push_back(phiOp);
    }
    return phiOps;
}

bool GimpleToPluginOps::IsDomInfoAvailable()
{
    return dom_info_available_p (CDI_DOMINATORS);
}

Value GimpleToPluginOps::CreateNewDefFor(uint64_t oldId, uint64_t opId, uint64_t defId)
{
    tree old_name = reinterpret_cast<tree>(oldId);
    gimple *stmt = reinterpret_cast<gimple*>(opId);
    tree ret;
    if (defId == 0) {
        gimple *stmt = reinterpret_cast<gimple*>(opId);
        ret = create_new_def_for(old_name, stmt, gimple_phi_result_ptr (stmt));
    } else {
        tree defTree = reinterpret_cast<tree>(defId);
        ret = create_new_def_for(old_name, stmt, &defTree);
    }
    uint64_t retId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(ret));
    return TreeToValue(retId);
}

mlir::Value GimpleToPluginOps::GetCurrentDefFor(uint64_t varId)
{
    tree var = reinterpret_cast<tree>(varId);
    tree ret = get_current_def(var);
    uint64_t retId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(ret));
    return TreeToValue(retId);
}

bool GimpleToPluginOps::SetCurrentDefFor(uint64_t varId, uint64_t defId)
{
    tree varTree = reinterpret_cast<tree>(varId);
    tree defTree = reinterpret_cast<tree>(defId);
    set_current_def(varTree, defTree);
    return true;
}

Value GimpleToPluginOps::CopySsaName(uint64_t id)
{
    tree ssaTree = reinterpret_cast<tree>(id);
    tree ret = copy_ssa_name(ssaTree);
    uint64_t retId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(ret));
    return TreeToValue(retId);
}

Value GimpleToPluginOps::MakeSsaName(mlir::Type type)
{
    PluginTypeBase pluginType = type.dyn_cast<PluginTypeBase>();
    uintptr_t typeId = pluginTypeTranslator.translateType(pluginType);
    tree treeType = reinterpret_cast<tree>(typeId);
    tree ret = make_ssa_name(treeType);
    uint64_t retId = reinterpret_cast<uint64_t>(
            reinterpret_cast<void*>(ret));
    return TreeToValue(retId);
}

} // namespace PluginIR