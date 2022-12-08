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

private:
    PluginIR::GimpleToPluginOps gimpleConversion;
}; // class PluginClientAPI
} // namespace PluginAPI

#endif // PLUGIN_OPS_CLIENT_API_H