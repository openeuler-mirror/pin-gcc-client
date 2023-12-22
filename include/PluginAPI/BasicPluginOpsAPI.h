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
    This file contains the declaration of the BasicPluginAPI class.
*/

#ifndef BASIC_PLUGIN_OPS_API_H
#define BASIC_PLUGIN_OPS_API_H

#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"

#include <vector>
#include <string>

namespace PluginAPI {
using std::vector;
using std::string;
using namespace mlir::Plugin;
using namespace PluginIR;

/* The BasicPluginAPI class defines the basic plugin API, both the plugin
   client and the server should inherit this class and implement there own
   defined API. */
class BasicPluginOpsAPI {
public:
    BasicPluginOpsAPI() = default;
    virtual ~BasicPluginOpsAPI() = default;

    virtual string GetDeclSourceFile(uint64_t gccDataAddr) = 0;
    virtual string VariableName(int64_t gccDataAddr) = 0;
    virtual string FuncName(int64_t gccDataAddr) = 0;
    virtual int GetDeclSourceLine(uint64_t gccDataAddr) = 0;
    virtual int GetDeclSourceColumn(uint64_t gccDataAddr) = 0;

    // CGnode
    virtual vector<uint64_t> GetCGnodeIDs() = 0;
    virtual CGnodeOp GetCGnodeOpById(uint64_t) = 0;

    virtual uint64_t CreateBlock(uint64_t, uint64_t) = 0;
    virtual void DeleteBlock(uint64_t, uint64_t) = 0;
    virtual void SetImmediateDominator(uint64_t, uint64_t, uint64_t) = 0;
    virtual uint64_t GetImmediateDominator(uint64_t, uint64_t) = 0;
    virtual uint64_t RecomputeDominator(uint64_t, uint64_t) = 0;
    virtual vector<FunctionOp> GetAllFunc() = 0;
    virtual vector<uint64_t> GetFunctions() = 0;
    virtual FunctionOp GetFunctionOpById(uint64_t) = 0;
    virtual vector<LocalDeclOp> GetDecls(uint64_t funcID) = 0;
    virtual vector<DeclBaseOp> GetFuncDecls(uint64_t funcID) = 0;
    virtual vector<FieldDeclOp> GetFields(uint64_t declID) = 0;
    virtual PluginIR::PluginTypeBase GetDeclType(uint64_t declID) = 0;
    virtual DeclBaseOp BuildDecl(IDefineCode, string, PluginTypeBase) = 0;

    virtual mlir::Value MakeNode(IDefineCode) = 0;
    virtual void SetDeclName(uint64_t, uint64_t) = 0;
    virtual void SetDeclType(uint64_t, uint64_t) = 0;
    virtual void SetDeclAlign(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetUserAlign(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetSourceLocation(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetAddressable(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetNonAddressablep(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetVolatile(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetDeclContext(uint64_t newfieldId, uint64_t declId) = 0;
    virtual void SetDeclChain(uint64_t newfieldId, uint64_t fieldId) = 0;

    virtual unsigned GetDeclTypeSize(uint64_t declId) = 0;

    virtual void SetTypeFields(uint64_t declId, uint64_t fieldId) = 0;
    virtual void LayoutType(uint64_t declId) = 0;
    virtual void LayoutDecl(uint64_t declId) = 0;

    virtual vector<LoopOp> GetLoopsFromFunc(uint64_t) = 0;
    virtual LoopOp GetLoopById(uint64_t) = 0;
    virtual bool IsBlockInside(uint64_t, uint64_t) = 0;
    virtual vector<uint64_t> GetBlocksInLoop(uint64_t) = 0;
    virtual void AddLoop(uint64_t, uint64_t, uint64_t) = 0;
    virtual uint64_t AllocateNewLoop(void) = 0;
    virtual void DeleteLoop(uint64_t) = 0;
    virtual void AddBlockToLoop(uint64_t, uint64_t) = 0;
    virtual uint64_t GetHeader(uint64_t) = 0;
    virtual uint64_t GetLatch(uint64_t) = 0;
    virtual void SetHeader(uint64_t, uint64_t) = 0;
    virtual void SetLatch(uint64_t, uint64_t) = 0;
    virtual vector<std::pair<uint64_t, uint64_t> > GetLoopExits(uint64_t) = 0;
    virtual std::pair<uint64_t, uint64_t> GetLoopSingleExit(uint64_t) = 0;
    virtual LoopOp GetBlockLoopFather(uint64_t) = 0;
    virtual LoopOp FindCommonLoop(uint64_t, uint64_t) = 0;
    virtual PhiOp GetPhiOp(uint64_t) = 0;
    virtual CallOp GetCallOp(uint64_t) = 0;
    virtual bool SetLhsInCallOp(uint64_t, uint64_t) = 0;
    virtual uint64_t CreateCallOp(uint64_t, uint64_t, vector<uint64_t> &) = 0;
    virtual uint64_t CreateCondOp(uint64_t, IComparisonCode, uint64_t, uint64_t, uint64_t, uint64_t) = 0;
    virtual void CreateFallthroughOp(uint64_t, uint64_t) = 0;
    virtual mlir::Value GetResultFromPhi(uint64_t) = 0;
    virtual mlir::Value GetCurrentDefFromSSA(uint64_t) = 0;
    virtual bool SetCurrentDefInSSA(uint64_t, uint64_t) = 0;
    virtual mlir::Value CopySSAOp(uint64_t) = 0;
    virtual mlir::Value CreateSSAOp(mlir::Type) = 0;

    virtual uint64_t CreateAssignOp(uint64_t, IExprCode, vector<uint64_t> &) = 0;
    virtual mlir::Value CreateConstOp(mlir::Attribute, mlir::Type) = 0;
    virtual uint32_t AddArgInPhiOp(uint64_t, uint64_t, uint64_t, uint64_t) = 0;
    virtual PhiOp CreatePhiOp(uint64_t, uint64_t) = 0;
    virtual bool UpdateSSA() = 0;
    virtual vector<PhiOp> GetPhiOpsInsideBlock(uint64_t bb) = 0;
    virtual vector<uint64_t> GetOpsInsideBlock(uint64_t bb) = 0;
    virtual bool IsDomInfoAvailable() = 0;
    virtual mlir::Value GetValue(uint64_t) = 0;
    virtual void DebugValue(uint64_t) = 0;
    virtual void DebugOperation(uint64_t) = 0;
    virtual void DebugBlock(uint64_t) = 0;
    virtual mlir::Value BuildMemRef(PluginTypeBase, uint64_t, uint64_t) = 0;
    virtual void RedirectFallthroughTarget(uint64_t, uint64_t) = 0;
    virtual void RemoveEdge(uint64_t, uint64_t) = 0;

    virtual bool IsLtoOptimize() = 0;
    virtual bool IsWholeProgram() = 0;

    virtual bool IsVirtualOperand(uint64_t) = 0;

    virtual void CalDominanceInfo(uint64_t, uint64_t) = 0;
    virtual vector<uint64_t> GetImmUseStmts(uint64_t) = 0;
    virtual mlir::Value GetGimpleVuse(uint64_t) = 0;
    virtual mlir::Value GetGimpleVdef(uint64_t) = 0;
    virtual vector<mlir::Value> GetSsaUseOperand(uint64_t) = 0;
    virtual vector<mlir::Value> GetSsaDefOperand(uint64_t) = 0;
    virtual vector<mlir::Value> GetPhiOrStmtUse(uint64_t) = 0;
    virtual vector<mlir::Value> GetPhiOrStmtDef(uint64_t) = 0;
    virtual bool RefsMayAlias(uint64_t, uint64_t, uint64_t) = 0;
    virtual bool PTIncludesDecl(uint64_t, uint64_t) = 0;
    virtual bool PTsIntersect(uint64_t, uint64_t) = 0;
}; // class BasicPluginOpsAPI
} // namespace PluginAPI

#endif // BASIC_PLUGIN_OPS_API_H