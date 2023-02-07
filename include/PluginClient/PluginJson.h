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
    This file contains the declaration of the PluginJson class.
    主要完成功能：将operation、Decl、Type、Inter、string等类型数据序列化
*/

#ifndef PLUGIN_JSON_H
#define PLUGIN_JSON_H

#include <string>
#include <vector>

#include "Dialect/PluginDialect.h"
#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"

namespace PinClient {
using std::string;
using std::vector;

class PluginJson {
public:
    void OpJsonSerialize(vector<mlir::Plugin::FunctionOp>& data, string& out);
    void LoopOpsJsonSerialize(vector<mlir::Plugin::LoopOp>& loops, string& out);
    void LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out);
    void BlocksJsonSerialize(vector<uint64_t>&, string&);
    void EdgesJsonSerialize(vector<std::pair<uint64_t, uint64_t> >&, string&);
    void EdgeJsonSerialize(std::pair<uint64_t, uint64_t>&, string&);
    void NopJsonSerialize(string&);
    void FunctionOpJsonSerialize(vector<mlir::Plugin::FunctionOp>& data, string& out);
    void LocalDeclsJsonSerialize(vector<mlir::Plugin::LocalDeclOp>& decls, string& out);
    void GetPhiOpsJsonSerialize(vector<mlir::Plugin::PhiOp> phiOps, string& out);
    Json::Value OperationJsonSerialize(mlir::Operation *, uint64_t&);
    Json::Value CallOpJsonSerialize(mlir::Plugin::CallOp& data);
    Json::Value CondOpJsonSerialize(mlir::Plugin::CondOp& data, uint64_t&);
    Json::Value PhiOpJsonSerialize(mlir::Plugin::PhiOp& data);
    Json::Value AssignOpJsonSerialize(mlir::Plugin::AssignOp& data);
    Json::Value BaseOpJsonSerialize(mlir::Plugin::BaseOp data);
    Json::Value FallThroughOpJsonSerialize(mlir::Plugin::FallThroughOp data, uint64_t&);
    Json::Value RetOpJsonSerialize(mlir::Plugin::RetOp data, uint64_t&);
    Json::Value ValueJsonSerialize(mlir::Value value);
    Json::Value MemOpJsonSerialize(mlir::Plugin::MemOp& data);
    Json::Value SSAOpJsonSerialize(mlir::Plugin::SSAOp& data);
    /* 将Type类型数据序列化 */
    Json::Value TypeJsonSerialize(PluginIR::PluginTypeBase& type);
    PluginIR::PluginTypeBase TypeJsonDeSerialize(const string& data, mlir::MLIRContext &context);
    /* 将整数型数据序列化 */
    void IntegerSerialize(int64_t data, string& out);
    /* 将字符串型数据序列化 */
    void StringSerialize(const string& data, string& out);
};
} // namespace PinClient
#endif
