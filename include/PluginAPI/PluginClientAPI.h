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

#ifndef PLUGIN_OPS_CLIENT_API_H
#define PLUGIN_OPS_CLIENT_API_H

#include "BasicPluginOpsAPI.h"
#include "Translate/GimpleToPluginOps.h"

namespace PluginAPI {
using std::vector;
using std::string;
using namespace mlir::Plugin;

class PluginClientAPI : public BasicPluginOpsAPI {
public:
    PluginClientAPI (mlir::MLIRContext &context) : gimpleConversion(context) {};
    PluginClientAPI () = default;
    ~PluginClientAPI () = default;

    uint64_t CreateBlock(uint64_t, uint64_t) override;
    void DeleteBlock(uint64_t, uint64_t) override;
    void SetImmediateDominator(uint64_t, uint64_t, uint64_t) override;
    uint64_t GetImmediateDominator(uint64_t, uint64_t) override;
    uint64_t RecomputeDominator(uint64_t, uint64_t) override;

    vector<FunctionOp> GetAllFunc() override;
    vector<LocalDeclOp> GetDecls(uint64_t funcID) override;
    vector<LoopOp> GetLoopsFromFunc(uint64_t) override;
    LoopOp GetLoopById(uint64_t) override;
    bool IsBlockInside(uint64_t, uint64_t) override;
    vector<uint64_t> GetBlocksInLoop(uint64_t) override;
    void AddLoop(uint64_t, uint64_t, uint64_t) override;
    uint64_t AllocateNewLoop(void) override;
    void DeleteLoop(uint64_t) override;
    uint64_t GetHeader(uint64_t) override;
    uint64_t GetLatch(uint64_t) override;
    vector<std::pair<uint64_t, uint64_t> > GetLoopExits(uint64_t) override;
    std::pair<uint64_t, uint64_t> GetLoopSingleExit(uint64_t) override;
    LoopOp GetBlockLoopFather(uint64_t) override;
    PhiOp GetPhiOp(uint64_t) override;
    CallOp GetCallOp(uint64_t) override;
    /* Plugin API for CallOp. */
    bool SetLhsInCallOp(uint64_t, uint64_t) override;
    uint64_t CreateCallOp(uint64_t, uint64_t, vector<uint64_t> &) override;
    /* Plugin API for CondOp. */
    uint64_t CreateCondOp(uint64_t, IComparisonCode, uint64_t, uint64_t, uint64_t, uint64_t) override;
    void CreateFallthroughOp(uint64_t, uint64_t) override;
    mlir::Value GetResultFromPhi(uint64_t) override;
    /* Plugin API for AssignOp. */
    uint64_t CreateAssignOp(uint64_t, IExprCode, vector<uint64_t> &) override;
    /* Plugin API for PhiOp. */
    bool AddArgInPhiOp(uint64_t, uint64_t, uint64_t, uint64_t) override;
    PhiOp CreatePhiOp(uint64_t, uint64_t) override;
    /* Plugin API for ConstOp. */
    mlir::Value CreateConstOp(mlir::Attribute, mlir::Type) override;
    bool UpdateSSA() override;
    vector<PhiOp> GetPhiOpsInsideBlock(uint64_t bb) override;
    bool IsDomInfoAvailable() override;
	mlir::Value GetValue(uint64_t) override;
	mlir::Value BuildMemRef(PluginIR::PluginTypeBase, uint64_t, uint64_t) override;

    mlir::Value GetCurrentDefFromSSA(uint64_t) override;
    bool SetCurrentDefInSSA(uint64_t, uint64_t) override;
    mlir::Value CopySSAOp(uint64_t) override;
    mlir::Value CreateSSAOp(mlir::Type) override;
    
    /* Plugin API for Control Flow. */
    mlir::Value CreateNewDef(uint64_t, uint64_t, uint64_t);

    void RedirectFallthroughTarget(uint64_t, uint64_t) override;
private:
    PluginIR::GimpleToPluginOps gimpleConversion;
}; // class PluginClientAPI
} // namespace PluginAPI

#endif // PLUGIN_OPS_CLIENT_API_H