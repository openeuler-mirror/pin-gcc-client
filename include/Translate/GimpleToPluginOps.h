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
using namespace mlir::Plugin;

namespace detail {
    class BlockFromGimpleTranslatorImpl;
    // class BlockToPluginIRTranslatorImpl;
} // namespace detail

class GimpleToPluginOps {
public:
    GimpleToPluginOps (mlir::MLIRContext &);
    ~GimpleToPluginOps ();

    /* ToPluginInterface */
    uint64_t CreateBlock(uint64_t, uint64_t);
    vector<mlir::Plugin::FunctionOp> GetAllFunction();
    vector<mlir::Plugin::LocalDeclOp> GetAllDecls(uint64_t);
    vector<mlir::Plugin::LoopOp> GetAllLoops(uint64_t);
    LoopOp GetLoop(uint64_t);
    bool IsBlockInside(uint64_t, uint64_t);
    vector<uint64_t> GetBlocksInLoop(uint64_t);
    uint64_t AllocateNewLoop(void);
    void DeleteLoop(uint64_t);
    void AddLoop (uint64_t, uint64_t, uint64_t);
    uint64_t GetHeader(uint64_t);
    uint64_t GetLatch(uint64_t);
    vector<std::pair<uint64_t, uint64_t> > GetLoopExits(uint64_t);
    std::pair<uint64_t, uint64_t> GetLoopSingleExit(uint64_t);
    LoopOp GetBlockLoopFather(uint64_t);
    CallOp BuildCallOp(uint64_t);
    bool SetGimpleCallLHS(uint64_t, uint64_t);
    uint64_t CreateGcond(IComparisonCode, uint64_t, uint64_t);
    FunctionOp BuildFunctionOp(uint64_t);
    Operation *BuildOperation(uint64_t);
    CondOp BuildCondOp(uint64_t, uint64_t, Block*, Block*, uint64_t, uint64_t);
    AssignOp BuildAssignOp(uint64_t);
    PhiOp BuildPhiOp(uint64_t);
    mlir::Value GetGphiResult(uint64_t);
    mlir::Value TreeToValue(uint64_t);

private:
    GimpleToPluginOps () = delete;
    mlir::OpBuilder builder;
    TypeFromPluginIRTranslator typeTranslator;

    // process basic_block
    std::unique_ptr<detail::BlockFromGimpleTranslatorImpl> bbTranslator;
    bool ProcessBasicBlock(intptr_t, Region&);
    bool ProcessGimpleStmt(intptr_t, Region&);
};
} // namespace PluginIR

#endif // GIMPLE_TO_PLUGINOPS_H
