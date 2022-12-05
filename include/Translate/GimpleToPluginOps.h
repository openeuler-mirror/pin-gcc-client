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

#include "Translate/ToPluginOpsInterface.h"
#include "Translate/TypeTranslation.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace PluginIR {
using std::vector;

class GimpleToPluginOps : public ToPluginOpsInterface {
public:
    GimpleToPluginOps (mlir::MLIRContext &context) : builder(&context) , typeTranslator(context) {}
    GimpleToPluginOps () = default;
    ~GimpleToPluginOps () = default;

    /* ToPluginInterface */
    vector<mlir::Plugin::FunctionOp> GetAllFunction() override;
    vector<mlir::Plugin::LocalDeclOp> GetAllDecls(uint64_t) override;
    vector<mlir::Plugin::LoopOp> GetAllLoops(uint64_t) override;
    LoopOp GetLoop(uint64_t) override;
    bool IsBlockInside(uint64_t, uint64_t) override;
    vector<uint64_t> GetBlocksInLoop(uint64_t) override;
    uint64_t AllocateNewLoop(void) override;
    void DeleteLoop(uint64_t) override;
    void AddLoop (uint64_t, uint64_t, uint64_t) override;
    uint64_t GetHeader(uint64_t) override;
    uint64_t GetLatch(uint64_t) override;
    vector<std::pair<uint64_t, uint64_t> > GetLoopExits(uint64_t) override;
    std::pair<uint64_t, uint64_t> GetLoopSingleExit(uint64_t) override;
    LoopOp GetBlockLoopFather(uint64_t) override;
private:
    mlir::OpBuilder builder;
    TypeFromPluginIRTranslator typeTranslator;
};
} // namespace PluginIR

#endif // GIMPLE_TO_PLUGINOPS_H