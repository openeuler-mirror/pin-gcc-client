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
    This file contains the implementation of the PluginJson class.
*/

#include <json/json.h>
#include "PluginAPI/PluginClientAPI.h"
#include "PluginClient/PluginLog.h"
#include "PluginClient/PluginJson.h"

namespace PinClient {
using std::map;
using namespace mlir::Plugin;
using namespace mlir;

static uintptr_t GetID(Json::Value node)
{
    string id = node.asString();
    return atol(id.c_str());
}

Json::Value PluginJson::TypeJsonSerialize(PluginIR::PluginTypeBase& type)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    uint64_t ReTypeId;
    uint64_t ReTypeWidth;

    ReTypeId = static_cast<uint64_t>(type.getPluginTypeID());
    item["id"] = std::to_string(ReTypeId);

    if (auto elemTy = type.dyn_cast<PluginIR::PluginPointerType>()) {
        auto baseTy = elemTy.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        item["elementType"] = TypeJsonSerialize(baseTy);
        if (elemTy.isReadOnlyElem()) {
            item["elemConst"] = "1";
        } else {
            item["elemConst"] = "0";
        }
    }

    if (type.getPluginIntOrFloatBitWidth() != 0) {
        ReTypeWidth = type.getPluginIntOrFloatBitWidth();
        item["width"] = std::to_string(ReTypeWidth);
    }

    if (type.isSignedPluginInteger()) {
        item["signed"] = "1";
    }

    if (type.isUnsignedPluginInteger()) {
        item["signed"] = "0";
    }

    root["type"] = item;
    return root;
}

PluginIR::PluginTypeBase PluginJson::TypeJsonDeSerialize(const string& data, mlir::MLIRContext &context)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    PluginIR::PluginTypeBase baseType;

    Json::Value type = root["type"];
    uint64_t id = GetID(type["id"]);
    PluginIR::PluginTypeID PluginTypeId = static_cast<PluginIR::PluginTypeID>(id);

    if (type["signed"] && (id >= static_cast<uint64_t>(PluginIR::UIntegerTy1ID)
        && id <= static_cast<uint64_t>(PluginIR::IntegerTy64ID))) {
        string s = type["signed"].asString();
        uint64_t width = GetID(type["width"]);
        if (s == "1") {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Signed);
        } else {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Unsigned);
        }
    } else if (type["width"] && (id == static_cast<uint64_t>(PluginIR::FloatTyID)
        || id == static_cast<uint64_t>(PluginIR::DoubleTyID))) {
        uint64_t width = GetID(type["width"]);
        baseType = PluginIR::PluginFloatType::get(&context, width);
    } else if (id == static_cast<uint64_t>(PluginIR::PointerTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString(), context);
        baseType = PluginIR::PluginPointerType::get(&context, elemTy, type["elemConst"].asString() == "1" ? 1 : 0);
    } else {
        if (PluginTypeId == PluginIR::VoidTyID) {
            baseType = PluginIR::PluginVoidType::get(&context);
        }
        if (PluginTypeId == PluginIR::BooleanTyID) {
            baseType = PluginIR::PluginBooleanType::get(&context);
        }
        if (PluginTypeId == PluginIR::UndefTyID) {
            baseType = PluginIR::PluginUndefType::get(&context);
        }
    }

    return baseType;
}

void PluginJson::FunctionOpJsonSerialize(vector<FunctionOp>& data, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    int i = 0;
    string operation;
    
    for (auto& d: data) {
        item["id"] = std::to_string(d.idAttr().getInt());
        if (d.declaredInlineAttr().getValue()) {
            item["attributes"]["declaredInline"] = "1";
        } else {
            item["attributes"]["declaredInline"] = "0";
        }
        item["attributes"]["funcName"] = d.funcNameAttr().getValue().str().c_str();
        auto &region = d.getRegion();
        size_t bbIdx = 0;
        for (auto &b : region) {
            string blockStr = "block" + std::to_string(bbIdx++);
            uint64_t bbAddress = 0;
            size_t opIdx = 0;
            for (auto &inst : b) {
                if (isa<PlaceholderOp>(inst)) {
                    continue;
                } else if (isa<SSAOp>(inst)) {
                    continue;
                } else if (isa<MemOp>(inst)) {
                    continue;
                } else if (isa<ConstOp>(inst)) {
                    continue;
                }
                string opStr = "Operation" + std::to_string(opIdx++);
                item["region"][blockStr]["ops"][opStr] = OperationJsonSerialize(&inst, bbAddress);
            }
            assert(bbAddress != 0);
            item["region"][blockStr]["address"] = std::to_string(bbAddress);
        }
        operation = "FunctionOp" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

Json::Value PluginJson::OperationJsonSerialize(mlir::Operation *operation, uint64_t &bbId)
{
    Json::Value root;
    if (AssignOp op = llvm::dyn_cast<AssignOp>(operation)) {
        root = AssignOpJsonSerialize(op);
    } else if (CallOp op = llvm::dyn_cast<CallOp>(operation)) {
        root = CallOpJsonSerialize(op);
    } else if (CondOp op = llvm::dyn_cast<CondOp>(operation)) {
        root = CondOpJsonSerialize(op, bbId);
    } else if (PhiOp op = llvm::dyn_cast<PhiOp>(operation)) {
        root = PhiOpJsonSerialize(op);
    } else if (FallThroughOp op = llvm::dyn_cast<FallThroughOp>(operation)) {
        root = FallThroughOpJsonSerialize(op, bbId);
    } else if (RetOp op = llvm::dyn_cast<RetOp>(operation)) {
        root = RetOpJsonSerialize(op, bbId);
    } else if (BaseOp op = llvm::dyn_cast<BaseOp>(operation)) {
        root = BaseOpJsonSerialize(op);
    } else if (DebugOp op = llvm::dyn_cast<DebugOp>(operation)) {
        root = DebugOpJsonSerialize(op);
    }
    root["OperationName"] = operation->getName().getStringRef().str();
    return root;
}

Json::Value PluginJson::BaseOpJsonSerialize(BaseOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["opCode"] = data.opCodeAttr().getValue().str().c_str();
    return root;
}

Json::Value PluginJson::RetOpJsonSerialize(RetOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    return root;
}

Json::Value PluginJson::DebugOpJsonSerialize(DebugOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    return root;
}

Json::Value PluginJson::FallThroughOpJsonSerialize(FallThroughOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["destaddr"] = std::to_string(data.destaddrAttr().getInt());
    return root;
}

void PluginJson::LocalDeclsJsonSerialize(vector<LocalDeclOp>& decls, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto& decl: decls) {
        item["id"] = std::to_string(decl.idAttr().getInt());
        item["attributes"]["symName"] = decl.symNameAttr().getValue().str().c_str();
        item["attributes"]["typeID"] = decl.typeIDAttr().getInt();
        item["attributes"]["typeWidth"] = decl.typeWidthAttr().getInt();
        operation = "localDecl" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::LoopOpsJsonSerialize(vector<mlir::Plugin::LoopOp>& loops, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto& loop: loops) {
        item["id"] = std::to_string(loop.idAttr().getInt());
        item["index"] = std::to_string(loop.indexAttr().getInt());
        item["attributes"]["innerLoopId"] = std::to_string(loop.innerLoopIdAttr().getInt());
        item["attributes"]["outerLoopId"] = std::to_string(loop.outerLoopIdAttr().getInt());
        item["attributes"]["numBlock"] = std::to_string(loop.numBlockAttr().getInt());
        operation = "loopOp" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out)
{
    Json::Value root;
    root["id"] = std::to_string(loop.idAttr().getInt());
    root["index"] = std::to_string(loop.indexAttr().getInt());
    root["attributes"]["innerLoopId"] = std::to_string(loop.innerLoopIdAttr().getInt());
    root["attributes"]["outerLoopId"] = std::to_string(loop.outerLoopIdAttr().getInt());
    root["attributes"]["numBlock"] = std::to_string(loop.numBlockAttr().getInt());
    out = root.toStyledString();
}

void PluginJson::BlocksJsonSerialize(vector<uint64_t>& blocks, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for (auto& block : blocks) {
        item["id"] = std::to_string(block);
        index = "block" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::EdgesJsonSerialize(vector<std::pair<uint64_t, uint64_t> >& edges, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for (auto& edge : edges) {
        item["src"] = std::to_string(edge.first);
        item["dest"] = std::to_string(edge.second);
        index = "edge" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::EdgeJsonSerialize(std::pair<uint64_t, uint64_t>& edge, string& out)
{
    Json::Value root;
    root["src"] = std::to_string(edge.first);
    root["dest"] = std::to_string(edge.second);
    out = root.toStyledString();
}

// void类型的Json序列化
void PluginJson::NopJsonSerialize(string& out)
{
    Json::Value root;
    out = root.toStyledString();
}

void PluginJson::GetPhiOpsJsonSerialize(vector<PhiOp> phiOps, string & out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string operation;
    uint64_t placeholder = 0;
    for (auto phi : phiOps) {
        item = OperationJsonSerialize(phi.getOperation(), placeholder);
        operation = "operation" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

Json::Value PluginJson::CallOpJsonSerialize(CallOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    Optional<StringRef> calleeName = data.callee();
    if (calleeName) {
        item["callee"] = calleeName->str();
    }
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.getArgOperands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::CondOpJsonSerialize(CondOp& data, uint64_t &bbId)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["condCode"] = std::to_string(data.condCodeAttr().getInt());
    item["lhs"] = ValueJsonSerialize(data.GetLHS());
    item["rhs"] = ValueJsonSerialize(data.GetRHS());
    bbId = data.addressAttr().getInt();
    item["address"] = std::to_string(bbId);
    item["tbaddr"] = std::to_string(data.tbaddrAttr().getInt());
    item["fbaddr"] = std::to_string(data.fbaddrAttr().getInt());
    return item;
}

Json::Value PluginJson::PhiOpJsonSerialize(PhiOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["capacity"] = std::to_string(data.capacityAttr().getInt());
    item["nArgs"] = std::to_string(data.nArgsAttr().getInt());
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::SSAOpJsonSerialize(SSAOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["defCode"] = std::to_string(data.defCodeAttr().getInt());
    item["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    item["nameVarId"] = std::to_string(data.nameVarIdAttr().getInt());
    item["ssaParmDecl"] = std::to_string(data.ssaParmDeclAttr().getInt());
    item["version"] = std::to_string(data.versionAttr().getInt());
    item["definingId"] = std::to_string(data.definingIdAttr().getInt());
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginJson::AssignOpJsonSerialize(AssignOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["exprCode"] = std::to_string(data.exprCodeAttr().getInt());
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::ValueJsonSerialize(mlir::Value data)
{
    Json::Value root;
    if (ConstOp cOp = data.getDefiningOp<ConstOp>()) {
        auto retTy = data.getType().dyn_cast<PluginIR::PluginTypeBase>();
        root["retType"] = TypeJsonSerialize(retTy);
        root["id"] = std::to_string(cOp.idAttr().getInt());
        root["defCode"] = std::to_string(static_cast<int32_t>(IDefineCode::IntCST));
        root["value"] = std::to_string(cOp.initAttr().cast<mlir::IntegerAttr>().getInt());
    } else if (MemOp mOp = data.getDefiningOp<MemOp>()) {
        root = MemOpJsonSerialize(mOp);
    } else if (SSAOp sOp = data.getDefiningOp<SSAOp>()) {
        root = SSAOpJsonSerialize(sOp);
    } else if (PlaceholderOp phOp = data.getDefiningOp<PlaceholderOp>()) {
        root["id"] = std::to_string(phOp.idAttr().getInt());
        root["defCode"] = std::to_string(phOp.defCodeAttr().getInt());
        auto retTy = phOp.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
        root["retType"] = TypeJsonSerialize(retTy);
    } else {
        LOGE("ERROR: Can't Serialize!");
    }
    return root;
}

Json::Value PluginJson::MemOpJsonSerialize(MemOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    mlir::Value base = data.GetBase();
    mlir::Value offset = data.GetOffset();
    root["base"] = ValueJsonSerialize(base);
    root["offset"] = ValueJsonSerialize(offset);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

void PluginJson::IntegerSerialize(int64_t data, string& out)
{
    Json::Value root;
    root["integerData"] = data;
    out = root.toStyledString();
}

void PluginJson::StringSerialize(const string& data, string& out)
{
    Json::Value root;
    root["stringData"] = data;
    out = root.toStyledString();
}
} // namespace PinClient
