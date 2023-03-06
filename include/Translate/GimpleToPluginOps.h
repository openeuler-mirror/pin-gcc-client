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
    This file contains the declaration of the GimpleToPlugin class.
*/

#ifndef GIMPLE_TO_PLUGINOPS_H
#define GIMPLE_TO_PLUGINOPS_H

#include "Dialect/PluginOps.h"
#include "Translate/TypeTranslation.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"

namespace PluginIR {
using std::vector;
using std::string;
using namespace mlir::Plugin;

namespace Detail {
    class BlockFromGimpleTranslatorImpl;
    // class BlockToPluginIRTranslatorImpl;
} // namespace Detail

class GimpleToPluginOps {
public:
    GimpleToPluginOps (mlir::MLIRContext &);
    ~GimpleToPluginOps ();

    string DeclSourceFile(uint64_t gccDataAddr);
    string GetVariableName(uint64_t gccDataAddr);
    string GetFuncName(uint64_t gccDataAddr);
    int DeclSourceLine(uint64_t gccDataAddr);
    int DeclSourceColumn(uint64_t gccDataAddr);
    /* ToPluginInterface */
    uint64_t CreateBlock(uint64_t, uint64_t);
    void DeleteBlock(uint64_t, uint64_t);
    void SetImmediateDominator(uint64_t, uint64_t, uint64_t);
    uint64_t GetImmediateDominator(uint64_t, uint64_t);
    uint64_t RecomputeDominator(uint64_t, uint64_t);

    // CGnode
    vector<uint64_t> GetCGnodeIDs();
    mlir::Plugin::CGnodeOp GetCGnodeOpById(uint64_t);
    CGnodeOp BuildCGnodeOp(uint64_t);
    bool IsRealSymbolOfCGnode(uint64_t);

    vector<mlir::Plugin::FunctionOp> GetAllFunction();
    vector<uint64_t> GetFunctionIDs();
    mlir::Plugin::FunctionOp GetFunctionById(uint64_t);
    vector<mlir::Plugin::LocalDeclOp> GetAllDecls(uint64_t);
    vector<mlir::Plugin::DeclBaseOp> GetFuncDecls(uint64_t);
    vector<FieldDeclOp> GetFields(uint64_t);
    PluginIR::PluginTypeBase GetDeclType(uint64_t);
    mlir::Plugin::DeclBaseOp BuildDecl(IDefineCode, string, PluginTypeBase);

    mlir::Value MakeNode(IDefineCode);
    void SetDeclName(uint64_t, uint64_t);
    void SetDeclType(uint64_t, uint64_t);
    void SetDeclAlign(uint64_t newfieldId, uint64_t fieldId);
    void SetUserAlign(uint64_t newfieldId, uint64_t fieldId);
    void SetSourceLocation(uint64_t newfieldId, uint64_t fieldId);
    void SetAddressable(uint64_t newfieldId, uint64_t fieldId);
    void SetNonAddressablep(uint64_t newfieldId, uint64_t fieldId);
    void SetVolatile(uint64_t newfieldId, uint64_t fieldId);
    void SetDeclContext(uint64_t newfieldId, uint64_t declId);
    void SetDeclChain(uint64_t newfieldId, uint64_t fieldId);

    unsigned GetDeclTypeSize(uint64_t declId);

    void SetTypeFields(uint64_t declId, uint64_t fieldId);
    void LayoutType(uint64_t declId);
    void LayoutDecl(uint64_t declId);

    vector<mlir::Plugin::LoopOp> GetAllLoops(uint64_t);
    LoopOp GetLoop(uint64_t);
    bool IsBlockInside(uint64_t, uint64_t);
    vector<uint64_t> GetBlocksInLoop(uint64_t);
    uint64_t AllocateNewLoop(void);
    void DeleteLoop(uint64_t);
    void AddLoop (uint64_t, uint64_t, uint64_t);
    void AddBlockToLoop(uint64_t, uint64_t);
    uint64_t GetHeader(uint64_t);
    uint64_t GetLatch(uint64_t);
    void SetHeader(uint64_t, uint64_t);
    void SetLatch(uint64_t, uint64_t);
    vector<std::pair<uint64_t, uint64_t> > GetLoopExits(uint64_t);
    std::pair<uint64_t, uint64_t> GetLoopSingleExit(uint64_t);
    LoopOp GetBlockLoopFather(uint64_t);
    CallOp BuildCallOp(uint64_t);
    bool SetGimpleCallLHS(uint64_t, uint64_t);
    uint32_t AddPhiArg(uint64_t, uint64_t, uint64_t, uint64_t);
    uint64_t CreateGcallVec(uint64_t, uint64_t, vector<uint64_t> &);
    uint64_t CreateGassign(uint64_t, IExprCode, vector<uint64_t> &);
    CondOp BuildCondOp(uint64_t, uint64_t, Block*, Block*, uint64_t, uint64_t);
    uint64_t CreateGphiNode(uint64_t, uint64_t);
    FunctionOp BuildFunctionOp(uint64_t);
    Operation *BuildOperation(uint64_t);
    uint64_t CreateGcond(uint64_t, IComparisonCode, uint64_t, uint64_t, uint64_t, uint64_t);
    void CreateFallthroughOp(uint64_t, uint64_t);
    AssignOp BuildAssignOp(uint64_t);
    PhiOp BuildPhiOp(uint64_t);
    AsmOp BuildAsmOp(uint64_t);
    ResxOp BuildResxOp(uint64_t);
    BindOp BuildBindOp(uint64_t);
    TryOp BuildTryOp(uint64_t);
    CatchOp BuildCatchOp(uint64_t);
    EHDispatchOp BuildEHDispatchOp(uint64_t);
    NopOp BuildNopOp(uint64_t);
    EHElseOp BuildEHElseOp(uint64_t);
    LabelOp BuildLabelOp(uint64_t);
    EHMntOp BuildEHMntOp(uint64_t);
    GotoOp BuildGotoOp(uint64_t, uint64_t, Block*, uint64_t);
    TransactionOp BuildTransactionOp(uint64_t);
    SwitchOp BuildSwitchOp(uint64_t);
    mlir::Value GetGphiResult(uint64_t);
    mlir::Value BuildIntCst(mlir::Type, int64_t);
    void GetTreeAttr(uint64_t, bool&, PluginTypeBase&);
    mlir::Value TreeToValue(uint64_t);
    void DebugValue(uint64_t);
    mlir::Value BuildMemRef(PluginIR::PluginTypeBase, uint64_t, uint64_t);
    bool UpdateSSA();
    vector<mlir::Plugin::PhiOp> GetPhiOpsInsideBlock(uint64_t);
    bool IsDomInfoAvailable();
    mlir::Value CreateNewDefFor(uint64_t, uint64_t, uint64_t);
    bool SetCurrentDefFor(uint64_t, uint64_t);
    mlir::Value GetCurrentDefFor(uint64_t);
    mlir::Value CopySsaName(uint64_t);
    mlir::Value MakeSsaName(mlir::Type);

    void RedirectFallthroughTarget(uint64_t, uint64_t);
    void RemoveEdge(uint64_t, uint64_t);

    bool IsLtoOptimize();
    bool IsWholeProgram();

private:
    GimpleToPluginOps () = delete;
    mlir::OpBuilder builder;
    TypeFromPluginIRTranslator typeTranslator;
    TypeToPluginIRTranslator pluginTypeTranslator;

    // process basic_block
    std::unique_ptr<Detail::BlockFromGimpleTranslatorImpl> bbTranslator;
    bool ProcessBasicBlock(intptr_t, Region&);
    bool ProcessGimpleStmt(intptr_t, Region&);
};
} // namespace PluginIR

#endif // GIMPLE_TO_PLUGINOPS_H
