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

#include "PluginAPI/PluginClientAPI.h"

namespace PluginAPI {
uint64_t PluginClientAPI::CreateBlock(uint64_t funcAddr, uint64_t bbAddr)
{
    return gimpleConversion.CreateBlock(funcAddr, bbAddr);
}

void PluginClientAPI::DeleteBlock(uint64_t funcAddr, uint64_t bbAddr)
{
    gimpleConversion.DeleteBlock(funcAddr, bbAddr);
}

void PluginClientAPI::SetImmediateDominator(uint64_t dir, uint64_t bbAddr,
                                            uint64_t domiAddr)
{
    gimpleConversion.SetImmediateDominator(dir, bbAddr, domiAddr);
}

uint64_t PluginClientAPI::GetImmediateDominator(uint64_t dir, uint64_t bbAddr)
{
    return gimpleConversion.GetImmediateDominator(dir, bbAddr);
}

uint64_t PluginClientAPI::RecomputeDominator(uint64_t dir, uint64_t bbAddr)
{
    return gimpleConversion.RecomputeDominator(dir, bbAddr);
}

vector<FunctionOp> PluginClientAPI::GetAllFunc()
{
    return gimpleConversion.GetAllFunction();
}

vector<LocalDeclOp> PluginClientAPI::GetDecls(uint64_t funcID)
{
    return gimpleConversion.GetAllDecls(funcID);
}

vector<LoopOp> PluginClientAPI::GetLoopsFromFunc(uint64_t funcID)
{
    return gimpleConversion.GetAllLoops(funcID);
}

LoopOp PluginClientAPI::GetLoopById(uint64_t loopID)
{
    return gimpleConversion.GetLoop(loopID);
}

bool PluginClientAPI::IsBlockInside(uint64_t loopID, uint64_t blockID)
{
    return gimpleConversion.IsBlockInside(loopID, blockID);
}

vector<uint64_t> PluginClientAPI::GetBlocksInLoop(uint64_t loopID)
{
    return gimpleConversion.GetBlocksInLoop(loopID);
}

void PluginClientAPI::AddLoop (uint64_t loop, uint64_t outer, uint64_t funcId)
{
    gimpleConversion.AddLoop(loop, outer, funcId);
}

uint64_t PluginClientAPI::AllocateNewLoop(void)
{
    return gimpleConversion.AllocateNewLoop();
}
void PluginClientAPI::DeleteLoop(uint64_t loopId)
{
    gimpleConversion.DeleteLoop(loopId);
}

uint64_t PluginClientAPI::GetHeader(uint64_t loopId)
{
    return gimpleConversion.GetHeader(loopId);
}

uint64_t PluginClientAPI::GetLatch(uint64_t loopId)
{
    return gimpleConversion.GetLatch(loopId);
}

vector<std::pair<uint64_t, uint64_t> > PluginClientAPI::GetLoopExits(uint64_t loopId)
{
    return gimpleConversion.GetLoopExits(loopId);
}
std::pair<uint64_t, uint64_t> PluginClientAPI::GetLoopSingleExit(uint64_t loopId)
{
    return gimpleConversion.GetLoopSingleExit(loopId);
}

LoopOp PluginClientAPI::GetBlockLoopFather(uint64_t blockId)
{
    return gimpleConversion.GetBlockLoopFather(blockId);
}
PhiOp PluginClientAPI::GetPhiOp(uint64_t id)
{
    return gimpleConversion.BuildPhiOp(id);
}

CallOp PluginClientAPI::GetCallOp(uint64_t id)
{
    return gimpleConversion.BuildCallOp(id);
}

bool PluginClientAPI::SetLhsInCallOp(uint64_t callId, uint64_t lhsId)
{
    return gimpleConversion.SetGimpleCallLHS(callId, lhsId);
}

bool PluginClientAPI::AddArgInPhiOp(uint64_t phiId, uint64_t argId,
                                    uint64_t predId, uint64_t succId)
{
    return gimpleConversion.AddPhiArg(phiId, argId, predId, succId);
}

uint64_t PluginClientAPI::CreateCallOp(uint64_t blockId, uint64_t funcId,
                                       vector<uint64_t> &argIds)
{
    return gimpleConversion.CreateGcallVec(blockId, funcId, argIds);
}

uint64_t PluginClientAPI::CreateAssignOp(uint64_t blockId, IExprCode iCode,
                                         vector<uint64_t> &argIds)
{
    return gimpleConversion.CreateGassign(blockId, iCode, argIds);
}

uint64_t PluginClientAPI::CreateCondOp(uint64_t blockId, IComparisonCode iCode,
                                       uint64_t LHS, uint64_t RHS)
{
    return gimpleConversion.CreateGcond(blockId, iCode, LHS, RHS);
}

mlir::Value PluginClientAPI::GetResultFromPhi(uint64_t id)
{
    return gimpleConversion.GetGphiResult(id);
}

PhiOp PluginClientAPI::CreatePhiOp(uint64_t argId, uint64_t blockId)
{
    uint64_t id = gimpleConversion.CreateGphiNode(argId, blockId);
    return gimpleConversion.BuildPhiOp(id);
}

bool PluginClientAPI::UpdateSSA()
{
    return gimpleConversion.UpdateSSA();
}

vector<mlir::Plugin::PhiOp> PluginClientAPI::GetPhiOpsInsideBlock(uint64_t bb)
{
    return gimpleConversion.GetPhiOpsInsideBlock(bb);
}

bool PluginClientAPI::IsDomInfoAvailable()
{
    return gimpleConversion.IsDomInfoAvailable();
}

} // namespace PluginAPI