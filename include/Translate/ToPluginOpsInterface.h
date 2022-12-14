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
    This file contains the declaration of the ToPluginInterface class
*/

#ifndef TO_PLUGINOPS_INTERFACE_H
#define TO_PLUGINOPS_INTERFACE_H

#include <vector>
#include "Dialect/PluginOps.h"

namespace PluginIR {
using std::vector;
using namespace mlir::Plugin;

/* The ToPluginInterface class defines the plugin interfaces that different
   compilers need to implement. */
class ToPluginOpsInterface {
public:
    /* Operation. */
    virtual vector<FunctionOp> GetAllFunction() = 0;
    virtual vector<LocalDeclOp> GetAllDecls(uint64_t) = 0;
    virtual vector<LoopOp> GetAllLoops(uint64_t) = 0;
    virtual LoopOp GetLoop(uint64_t) = 0;
    virtual bool IsBlockInside(uint64_t, uint64_t) = 0;
    virtual vector<uint64_t> GetBlocksInLoop(uint64_t)  = 0;
    virtual uint64_t AllocateNewLoop(void) = 0;
    virtual void DeleteLoop(uint64_t) = 0;
    virtual void AddLoop (uint64_t, uint64_t, uint64_t) = 0;
    virtual uint64_t GetHeader(uint64_t) = 0;
    virtual uint64_t GetLatch(uint64_t) = 0;
    virtual vector<std::pair<uint64_t, uint64_t> > GetLoopExits(uint64_t) = 0;
    virtual std::pair<uint64_t, uint64_t> GetLoopSingleExit(uint64_t) = 0;
    virtual LoopOp GetBlockLoopFather(uint64_t) = 0;
    virtual bool UpdateSSA() = 0;
    virtual vector<mlir::Plugin::PhiOp> GetPhiOpsInsideBlock(uint64_t) = 0;
    virtual bool IsDomInfoAvailable() = 0;

};
} // namespace PluginIR

#endif // TO_PLUGINOPS_INTERFACE_H