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
    // CGnodeOp
    void CGnodeOpJsonSerialize(mlir::Plugin::CGnodeOp& cgnode, string& out);

    void OpJsonSerialize(vector<mlir::Plugin::FunctionOp>& data, string& out);
    void LoopOpsJsonSerialize(vector<mlir::Plugin::LoopOp>& loops, string& out);
    void LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out);
    void IDsJsonSerialize(vector<uint64_t>&, string&);
    void EdgesJsonSerialize(vector<std::pair<uint64_t, uint64_t> >&, string&);
    void EdgeJsonSerialize(std::pair<uint64_t, uint64_t>&, string&);
    void NopJsonSerialize(string&);
    void FunctionOpJsonSerialize(vector<mlir::Plugin::FunctionOp>& data, string& out);
    void LocalDeclsJsonSerialize(vector<mlir::Plugin::LocalDeclOp>& decls, string& out);
    void FunctionDeclsJsonSerialize(vector<mlir::Plugin::DeclBaseOp>& decls, string& out);
    void FiledOpsJsonSerialize(vector<mlir::Plugin::FieldDeclOp>& decls, string& out);
    void GetPhiOpsJsonSerialize(vector<mlir::Plugin::PhiOp> phiOps, string& out);
    void OpsJsonSerialize(vector<std::pair<mlir::Operation*, uint64_t>>& ops, string& out);
    void ValuesJsonSerialize(vector<mlir::Value>& values, string& out);
    Json::Value OperationJsonSerialize(mlir::Operation*, uint64_t&);
    Json::Value CallOpJsonSerialize(mlir::Plugin::CallOp& data);
    Json::Value CondOpJsonSerialize(mlir::Plugin::CondOp& data, uint64_t&);
    Json::Value PhiOpJsonSerialize(mlir::Plugin::PhiOp& data);
    Json::Value AssignOpJsonSerialize(mlir::Plugin::AssignOp& data);
    Json::Value BaseOpJsonSerialize(mlir::Plugin::BaseOp data);
    Json::Value DebugOpJsonSerialize(mlir::Plugin::DebugOp data);
    Json::Value FallThroughOpJsonSerialize(mlir::Plugin::FallThroughOp data, uint64_t&);
    Json::Value RetOpJsonSerialize(mlir::Plugin::RetOp data, uint64_t&);
    Json::Value ValueJsonSerialize(mlir::Value value);
    Json::Value MemOpJsonSerialize(mlir::Plugin::MemOp& data);
    Json::Value SSAOpJsonSerialize(mlir::Plugin::SSAOp& data);
    Json::Value AsmOpJsonSerialize(mlir::Plugin::AsmOp data);
    Json::Value SwitchOpJsonSerialize(mlir::Plugin::SwitchOp data, uint64_t &bbId);
    Json::Value ListOpJsonSerialize(mlir::Plugin::ListOp& data);
    Json::Value StrOpJsonSerialize(mlir::Plugin::StrOp& data);
    Json::Value ArrayOpJsonSerialize(mlir::Plugin::ArrayOp& data);
    Json::Value DeclBaseOpJsonSerialize(mlir::Plugin::DeclBaseOp& data);
    Json::Value BlockOpJsonSerialize(mlir::Plugin::BlockOp& data);
    Json::Value VecOpJsonSerialize(mlir::Plugin::VecOp& data);
    Json::Value FieldDeclOpJsonSerialize(mlir::Plugin::FieldDeclOp& data);
    Json::Value ConstructorOpJsonSerialize(mlir::Plugin::ConstructorOp& data);
    Json::Value ComponentOpJsonSerialize(mlir::Plugin::ComponentOp& data);
    Json::Value GotoOpJsonSerialize(mlir::Plugin::GotoOp data, uint64_t&);
    Json::Value AddressOpJsonSerialize(mlir::Plugin::AddressOp& data);
    Json::Value TransactionOpJsonSerialize(mlir::Plugin::TransactionOp data, uint64_t&);
    Json::Value LabelOpJsonSerialize(mlir::Plugin::LabelOp data);
    Json::Value EHMntOpJsonSerialize(mlir::Plugin::EHMntOp data);
    Json::Value ResxOpJsonSerialize(mlir::Plugin::ResxOp data, uint64_t&);
    Json::Value BindOpJsonSerialize(mlir::Plugin::BindOp data);
    Json::Value TryOpJsonSerialize(mlir::Plugin::TryOp data);
    Json::Value CatchOpJsonSerialize(mlir::Plugin::CatchOp data);
    Json::Value EHDispatchOpJsonSerialize(mlir::Plugin::EHDispatchOp data, uint64_t&);
    Json::Value NopOpJsonSerialize(mlir::Plugin::NopOp& data);
    Json::Value EHElseOpJsonSerialize(mlir::Plugin::EHElseOp& data);
    /* 将Type类型数据序列化 */
    Json::Value TypeJsonSerialize(PluginIR::PluginTypeBase type);
    PluginIR::PluginTypeBase TypeJsonDeSerialize(const string& data, mlir::MLIRContext &context);
    /* 将整数型数据序列化 */
    void IntegerSerialize(int64_t data, string& out);
    /* 将字符串型数据序列化 */
    void StringSerialize(const string& data, string& out);
};
} // namespace PinClient
#endif
