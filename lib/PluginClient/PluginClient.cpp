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
    This file contains the implementation of the PluginClient class..
*/
#include "Dialect/PluginDialect.h"
#include "Dialect/PluginTypes.h"
#include "PluginAPI/PluginClientAPI.h"
#include "IRTrans/IRTransPlugin.h"

#include <thread>
#include <fstream>
#include <iostream>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <unistd.h>
#include <json/json.h>

namespace PinClient {
using namespace mlir::Plugin;
using namespace mlir;
using std::ios;
static std::shared_ptr<PluginClient> g_plugin = nullptr;
const char *g_portFilePath = "/tmp/grpc_ports_pin_client.txt";
static std::shared_ptr<Channel> g_grpcChannel = nullptr;

std::map<InjectPoint, plugin_event> g_injectPoint {
    {HANDLE_PARSE_TYPE, PLUGIN_FINISH_TYPE},
    {HANDLE_PARSE_DECL, PLUGIN_FINISH_DECL},
    {HANDLE_PRAGMAS, PLUGIN_PRAGMAS},
    {HANDLE_PARSE_FUNCTION, PLUGIN_FINISH_PARSE_FUNCTION},
    {HANDLE_BEFORE_IPA, PLUGIN_ALL_IPA_PASSES_START},
    {HANDLE_AFTER_IPA, PLUGIN_ALL_IPA_PASSES_END},
    {HANDLE_BEFORE_EVERY_PASS, PLUGIN_EARLY_GIMPLE_PASSES_START},
    {HANDLE_AFTER_EVERY_PASS, PLUGIN_EARLY_GIMPLE_PASSES_END},
    {HANDLE_BEFORE_ALL_PASS, PLUGIN_ALL_PASSES_START},
    {HANDLE_AFTER_ALL_PASS, PLUGIN_ALL_PASSES_END},
    {HANDLE_COMPILE_END, PLUGIN_FINISH},
    {HANDLE_MANAGER_SETUP, PLUGIN_PASS_MANAGER_SETUP},
};

std::shared_ptr<PluginClient> PluginClient::GetInstance(void)
{
    return g_plugin;
}

int PluginClient::GetEvent(InjectPoint inject, plugin_event *event)
{
    auto it = g_injectPoint.find(inject);
    if (it != g_injectPoint.end()) {
        *event = it->second;
        return 0;
    }
    return -1;
}

static uintptr_t GetID(Json::Value node)
{
    string id = node.asString();
    return atol(id.c_str());
}

Json::Value PluginClient::TypeJsonSerialize (PluginIR::PluginTypeBase& type)
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
        }else {
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

PluginIR::PluginTypeBase PluginClient::TypeJsonDeSerialize(const string& data, mlir::MLIRContext &context)
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
        }
        else {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Unsigned);
        }
    }
    else if (type["width"] && (id == static_cast<uint64_t>(PluginIR::FloatTyID)
             || id == static_cast<uint64_t>(PluginIR::DoubleTyID)) ) {
        uint64_t width = GetID(type["width"]);
        baseType = PluginIR::PluginFloatType::get(&context, width);
    }else if (id == static_cast<uint64_t>(PluginIR::PointerTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString(), context);
        auto ty = elemTy.dyn_cast<PluginIR::PluginTypeBase>();
        baseType = PluginIR::PluginPointerType::get(&context, elemTy, type["elemConst"].asString() == "1" ? 1 : 0);
    }else {
        if (PluginTypeId == PluginIR::VoidTyID)
            baseType = PluginIR::PluginVoidType::get(&context);
        if (PluginTypeId == PluginIR::BooleanTyID)
            baseType = PluginIR::PluginBooleanType::get(&context);
        if (PluginTypeId == PluginIR::UndefTyID)
            baseType = PluginIR::PluginUndefType::get(&context);
    }

    return baseType;
}

void PluginClient::FunctionOpJsonSerialize(vector<FunctionOp>& data, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    int i = 0;
    string operation;
    
    for (auto& d: data) {
        item["id"] = std::to_string(d.idAttr().getInt());
        // item["opCode"] = OP_FUNCTION;
        if (d.declaredInlineAttr().getValue())
            item["attributes"]["declaredInline"] = "1";
        else
            item["attributes"]["declaredInline"] = "0";
        item["attributes"]["funcName"] = d.funcNameAttr().getValue().str().c_str();
        auto &region = d.getRegion();
        size_t bbIdx = 0;
        for (auto &b : region) {
            string blockStr = "block" + std::to_string(bbIdx++);
            uint64_t bbAddress = 0;
            size_t opIdx = 0;
            for (auto &inst : b) {
                if (auto phOp = llvm::dyn_cast<PlaceholderOp>(inst)) continue;
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

Json::Value PluginClient::OperationJsonSerialize(mlir::Operation *operation,
                                                 uint64_t &bbId)
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
    }
    root["OperationName"] = operation->getName().getStringRef().str();
    return root;
}

Json::Value PluginClient::BaseOpJsonSerialize(BaseOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["opCode"] = data.opCodeAttr().getValue().str().c_str();
    return root;
}

Json::Value PluginClient::RetOpJsonSerialize(RetOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    return root;
}

Json::Value PluginClient::FallThroughOpJsonSerialize(FallThroughOp data,
                                                     uint64_t &bbId)
{
    Json::Value root;
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["destaddr"] = std::to_string(data.destaddrAttr().getInt());
    return root;
}

void PluginClient::LocalDeclsJsonSerialize(vector<LocalDeclOp>& decls, string& out)
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
        operation = "operation" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginClient::LoopOpsJsonSerialize(vector<mlir::Plugin::LoopOp>& loops, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto&loop: loops) {
        item["id"] = std::to_string(loop.idAttr().getInt());
        item["index"] = std::to_string(loop.indexAttr().getInt());
        item["attributes"]["innerLoopId"] = std::to_string(loop.innerLoopIdAttr().getInt());
        item["attributes"]["outerLoopId"] = std::to_string(loop.outerLoopIdAttr().getInt());
        item["attributes"]["numBlock"] = std::to_string(loop.numBlockAttr().getInt());
        operation = "operation" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginClient::LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out)
{
    Json::Value root;
    root["id"] = std::to_string(loop.idAttr().getInt());
    root["index"] = std::to_string(loop.indexAttr().getInt());
    root["attributes"]["innerLoopId"] = std::to_string(loop.innerLoopIdAttr().getInt());
    root["attributes"]["outerLoopId"] = std::to_string(loop.outerLoopIdAttr().getInt());
    root["attributes"]["numBlock"] = std::to_string(loop.numBlockAttr().getInt());
    out = root.toStyledString();
}

void PluginClient::BlocksJsonSerialize(vector<uint64_t>& blocks, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for(auto& block : blocks) {
        item["id"] = std::to_string(block);
        index = "block" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginClient::EdgesJsonSerialize(vector<pair<uint64_t, uint64_t> >& edges, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for(auto& edge : edges) {
        item["src"] = std::to_string(edge.first);
        item["dest"] = std::to_string(edge.second);
        index = "edge" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginClient::EdgeJsonSerialize(pair<uint64_t, uint64_t>& edge, string& out)
{
    Json::Value root;
    root["src"] = std::to_string(edge.first);
    root["dest"] = std::to_string(edge.second);
    out = root.toStyledString();
}

// void类型的Json序列化
void PluginClient::NopJsonSerialize(string& out)
{
    Json::Value root;
    out = root.toStyledString();
}

void PluginClient::GetPhiOpsJsonSerialize(vector<PhiOp> phiOps, string & out)
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

Json::Value PluginClient::CallOpJsonSerialize(CallOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["callee"] = data.callee().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.getArgOperands()) {
        PlaceholderOp phOp = v.getDefiningOp<PlaceholderOp>();
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input]["id"] = std::to_string(phOp.idAttr().getInt());
        item["operands"][input]["defCode"] = std::to_string(phOp.defCodeAttr().getInt());
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginClient::CondOpJsonSerialize(CondOp& data, uint64_t &bbId)
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

Json::Value PluginClient::PhiOpJsonSerialize(PhiOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["capacity"] = std::to_string(data.capacityAttr().getInt());
    item["nArgs"] = std::to_string(data.nArgsAttr().getInt());
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
        PlaceholderOp phOp = v.getDefiningOp<PlaceholderOp>();
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input]["id"] = std::to_string(phOp.idAttr().getInt());
        item["operands"][input]["defCode"] = std::to_string(phOp.defCodeAttr().getInt());
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginClient::AssignOpJsonSerialize(AssignOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["exprCode"] = std::to_string(data.exprCodeAttr().getInt());
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
        PlaceholderOp phOp = v.getDefiningOp<PlaceholderOp>();
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input]["id"] = std::to_string(phOp.idAttr().getInt());
        item["operands"][input]["defCode"] = std::to_string(phOp.defCodeAttr().getInt());
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginClient::ValueJsonSerialize(mlir::Value data)
{
    Json::Value root;
    if (PlaceholderOp phOp = data.getDefiningOp<PlaceholderOp>()) {
        root["id"] = std::to_string(phOp.idAttr().getInt());
        root["defCode"] = std::to_string(phOp.defCodeAttr().getInt());
        auto retTy = phOp.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
        root["retType"] = TypeJsonSerialize(retTy);
    } else {
        LOGE("ERROR: Can't Serialize!");
    }
    return root;
}

Json::Value PluginClient::MemOpJsonSerialize(MemOp& data)
{
    Json::Value root; 
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    mlir::Value base = data.GetBase();
    mlir::Value offset = data.GetOffset();
    root["base"] = ValueJsonSerialize(base);
    root["offset"] = ValueJsonSerialize(offset);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

void PluginClient::IRTransBegin(const string& funcName, const string& param)
{
    string result;
    
    Json::Value root;
    Json::Reader reader;
    reader.parse(param, root);
    LOGD("%s func:%s,param:%s\n", __func__, funcName.c_str(), param.c_str());

    if (funcName == "GetAllFunc") {
        // Load our Dialect in this MLIR Context.
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        vector<FunctionOp> allFuncOps = clientAPI.GetAllFunc();
        FunctionOpJsonSerialize(allFuncOps, result);
        this->ReceiveSendMsg("FuncOpResult", result);
    } else if (funcName == "GetLocalDecls") {
        /// Json格式
        /// {
        ///     "funcId":"xxxx"
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string funcIdKey = "funcId";
        uint64_t funcID = atol(root[funcIdKey].asString().c_str());
        vector<LocalDeclOp> decls = clientAPI.GetDecls(funcID);
        LocalDeclsJsonSerialize(decls, result);
        this->ReceiveSendMsg("LocalDeclOpResult", result);
    } else if (funcName == "GetLoopsFromFunc") {
        /// Json格式
        /// {
        ///     "funcId":"xxxx"
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string funcIdKey = "funcId";
        uint64_t funcID = atol(root[funcIdKey].asString().c_str());
        vector<LoopOp> irLoops = clientAPI.GetLoopsFromFunc(funcID);
        LoopOpsJsonSerialize(irLoops, result);
        this->ReceiveSendMsg("LoopOpsResult", result);
    } else if (funcName == "GetLoopById") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx"
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        LoopOp irLoop = clientAPI.GetLoopById(loopId);
        LoopOpJsonSerialize(irLoop, result);
        this->ReceiveSendMsg("LoopOpResult", result);
    } else if (funcName == "IsBlockInside") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        ///     "blockId":"xxxx"
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        std::string blockIdKey = "blockId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        uint64_t blockId = atol(root[blockIdKey].asString().c_str());
        bool res = clientAPI.IsBlockInside(loopId, blockId);
        this->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)res));
    } else if (funcName == "AllocateNewLoop") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t newLoopId = clientAPI.AllocateNewLoop();
        LoopOp newLoop = clientAPI.GetLoopById(newLoopId);
        LoopOpJsonSerialize(newLoop, result);
        this->ReceiveSendMsg("LoopOpResult", result);
    } else if (funcName == "DeleteLoop") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        clientAPI.DeleteLoop(loopId);
        NopJsonSerialize(result);
        this->ReceiveSendMsg("VoidResult", result);
    } else if (funcName == "AddLoop") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        ///     "outerId":"xxxx"
        ///     "funcId":"xxxx"
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        std::string outerIdKey = "outerId";
        std::string funcIdKey = "funcId";
        uint64_t loopID = atol(root[loopIdKey].asString().c_str());
        uint64_t outerID = atol(root[outerIdKey].asString().c_str());
        uint64_t funcID = atol(root[funcIdKey].asString().c_str());
        clientAPI.AddLoop(loopID, outerID, funcID);
        LoopOp irLoop = clientAPI.GetLoopById(loopID);
        LoopOpJsonSerialize(irLoop, result);
        this->ReceiveSendMsg("LoopOpResult", result);
    } else if (funcName == "GetBlocksInLoop") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        vector<uint64_t> blocks = clientAPI.GetBlocksInLoop(loopId);
        BlocksJsonSerialize(blocks, result);
        this->ReceiveSendMsg("IdsResult", result);
    } else if (funcName == "GetHeader") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        uint64_t blockId = clientAPI.GetHeader(loopId);
        this->ReceiveSendMsg("IdResult", std::to_string(blockId));
    } else if (funcName == "GetLatch") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        uint64_t blockId = clientAPI.GetLatch(loopId);
        this->ReceiveSendMsg("IdResult", std::to_string(blockId));
    } else if (funcName == "GetLoopExits") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        vector<pair<uint64_t, uint64_t> > edges = clientAPI.GetLoopExits(loopId);
        EdgesJsonSerialize(edges, result);
        this->ReceiveSendMsg("EdgesResult", result);
    } else if (funcName == "GetLoopSingleExit") {
        /// Json格式
        /// {
        ///     "loopId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string loopIdKey = "loopId";
        uint64_t loopId = atol(root[loopIdKey].asString().c_str());
        pair<uint64_t, uint64_t> edge = clientAPI.GetLoopSingleExit(loopId);
        EdgeJsonSerialize(edge, result);
        this->ReceiveSendMsg("EdgeResult", result);
    } else if (funcName == "GetBlockLoopFather") {
        /// Json格式
        /// {
        ///     "blockId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string blockIdKey = "blockId";
        uint64_t blockId = atol(root[blockIdKey].asString().c_str());
        LoopOp loopFather = clientAPI.GetBlockLoopFather(blockId);
        LoopOpJsonSerialize(loopFather, result);
        this->ReceiveSendMsg("LoopOpResult", result);
    } else if (funcName == "CreateBlock") {
        /// Json格式
        /// {
        ///     "bbaddr":"xxxx",
        ///     "funcaddr":"xxxx"
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t blockAddr = atol(root["bbaddr"].asString().c_str());
        uint64_t funcAddr = atol(root["funcaddr"].asString().c_str());
        uint64_t newBBAddr = clientAPI.CreateBlock(funcAddr, blockAddr);
        this->ReceiveSendMsg("IdResult", std::to_string(newBBAddr));
    } else if (funcName == "DeleteBlock") {
        /// Json格式
        /// {
        ///     "bbaddr":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string BlockIdKey = "bbaddr";
        uint64_t addr = atol(root[BlockIdKey].asString().c_str());
        clientAPI.DeleteLoop(addr);
        NopJsonSerialize(result);
        this->ReceiveSendMsg("VoidResult", result);
    } else if (funcName == "SetImmediateDominator") {
        /// Json格式
        /// {
        ///     "dir":"xxxx",
        ///     "bbaddr":"xxxx",
        ///     "domiaddr":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string dirIdKey = "dir";
        uint64_t dir = atol(root[dirIdKey].asString().c_str());
        std::string BlockIdKey = "bbaddr";
        uint64_t bbaddr = atol(root[BlockIdKey].asString().c_str());
        std::string domiIdKey = "domiaddr";
        uint64_t domiaddr = atol(root[domiIdKey].asString().c_str());
        clientAPI.SetImmediateDominator(dir, bbaddr, domiaddr);
        NopJsonSerialize(result);
        this->ReceiveSendMsg("VoidResult", result);
    } else if (funcName == "GetImmediateDominator") {
        /// Json格式
        /// {
        ///     "dir":"xxxx",
        ///     "bbaddr":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string dirIdKey = "dir";
        uint64_t dir = atol(root[dirIdKey].asString().c_str());
        std::string BlockIdKey = "bbaddr";
        uint64_t bbaddr = atol(root[BlockIdKey].asString().c_str());
        uint64_t ret = clientAPI.GetImmediateDominator(dir, bbaddr);
        this->ReceiveSendMsg("IdResult", std::to_string(ret));
    } else if (funcName == "RecomputeDominator") {
        /// Json格式
        /// {
        ///     "dir":"xxxx",
        ///     "bbaddr":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string dirIdKey = "dir";
        uint64_t dir = atol(root[dirIdKey].asString().c_str());
        std::string BlockIdKey = "bbaddr";
        uint64_t bbaddr = atol(root[BlockIdKey].asString().c_str());
        uint64_t ret = clientAPI.RecomputeDominator(dir, bbaddr);
        this->ReceiveSendMsg("IdResult", std::to_string(ret));
    } else if (funcName == "GetPhiOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t id = atol(root[std::to_string(0)].asString().c_str());
        PhiOp op = clientAPI.GetPhiOp(id);
        Json::Value result = PhiOpJsonSerialize(op);
        this->ReceiveSendMsg("OpsResult", result.toStyledString());
    } else if (funcName == "GetCallOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t id = atol(root[std::to_string(0)].asString().c_str());
        CallOp op = clientAPI.GetCallOp(id);
        Json::Value result = CallOpJsonSerialize(op);
        this->ReceiveSendMsg("OpsResult", result.toStyledString());
    } else if (funcName == "SetLhsInCallOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t callId = atol(root["callId"].asString().c_str());
        uint64_t lhsId = atol(root["lhsId"].asString().c_str());
        bool ret = clientAPI.SetLhsInCallOp(callId, lhsId);
        this->ReceiveSendMsg("BoolResult", std::to_string(ret));
    } else if (funcName == "AddArgInPhiOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t phiId = atol(root["phiId"].asString().c_str());
        uint64_t argId = atol(root["argId"].asString().c_str());
        uint64_t predId = atol(root["predId"].asString().c_str());
        uint64_t succId = atol(root["succId"].asString().c_str());
        bool ret = clientAPI.AddArgInPhiOp(phiId, argId, predId, succId);
        this->ReceiveSendMsg("BoolResult", std::to_string(ret));
    } else if (funcName == "CreateCallOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t blockId = atol(root["blockId"].asString().c_str());
        uint64_t funcId = atol(root["funcId"].asString().c_str());
        vector<uint64_t> argIds;
        Json::Value argIdsJson = root["argIds"];
        Json::Value::Members member = argIdsJson.getMemberNames();
        for (Json::Value::Members::iterator opIter = member.begin();
             opIter != member.end(); opIter++) {
            string key = *opIter;
            uint64_t id = atol(argIdsJson[key.c_str()].asString().c_str());
            argIds.push_back(id);
        }
        uint64_t ret = clientAPI.CreateCallOp(blockId, funcId, argIds);
        this->ReceiveSendMsg("IdResult", std::to_string(ret));
    } else if (funcName == "CreateAssignOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t blockId = atol(root["blockId"].asString().c_str());
        int condCode = atol(root["exprCode"].asString().c_str());
        vector<uint64_t> argIds;
        Json::Value argIdsJson = root["argIds"];
        Json::Value::Members member = argIdsJson.getMemberNames();
        for (Json::Value::Members::iterator opIter = member.begin();
             opIter != member.end(); opIter++) {
            string key = *opIter;
            uint64_t id = atol(argIdsJson[key.c_str()].asString().c_str());
            argIds.push_back(id);
        }
        uint64_t ret = clientAPI.CreateAssignOp(
                blockId, IExprCode(condCode), argIds);
        this->ReceiveSendMsg("IdResult", std::to_string(ret));
    } else if (funcName == "CreateCondOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t blockId = atol(root["blockId"].asString().c_str());
        int condCode = atol(root["condCode"].asString().c_str());
        uint64_t lhsId = atol(root["lhsId"].asString().c_str());
        uint64_t rhsId = atol(root["rhsId"].asString().c_str());
        uint64_t tbaddr = atol(root["tbaddr"].asString().c_str());
        uint64_t fbaddr = atol(root["fbaddr"].asString().c_str());
        uint64_t ret = clientAPI.CreateCondOp(blockId, IComparisonCode(condCode),
                                              lhsId, rhsId, tbaddr, fbaddr);
        this->ReceiveSendMsg("IdResult", std::to_string(ret));
    } else if (funcName == "CreateFallthroughOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t address = atol(root["address"].asString().c_str());
        uint64_t destaddr = atol(root["destaddr"].asString().c_str());
        clientAPI.CreateFallthroughOp(address, destaddr);
        NopJsonSerialize(result);
        this->ReceiveSendMsg("VoidResult", result);
    } else if (funcName == "GetResultFromPhi") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t id = atol(root["id"].asString().c_str());
        mlir::Value ret = clientAPI.GetResultFromPhi(id);
        this->ReceiveSendMsg("ValueResult",
                             ValueJsonSerialize(ret).toStyledString());
    } else if (funcName == "CreatePhiOp") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t argId = atol(root["argId"].asString().c_str());
        uint64_t blockId = atol(root["blockId"].asString().c_str());
        PhiOp op = clientAPI.CreatePhiOp(argId, blockId);
        Json::Value result = PhiOpJsonSerialize(op);
        this->ReceiveSendMsg("OpsResult", result.toStyledString());
    } else if (funcName == "UpdateSSA") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        bool ret = clientAPI.UpdateSSA();
        this->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)ret));
    } else if (funcName == "GetAllPhiOpInsideBlock") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t bb = atol(root["bbAddr"].asString().c_str());
        vector<PhiOp> phiOps = clientAPI.GetPhiOpsInsideBlock(bb);
        GetPhiOpsJsonSerialize(phiOps, result);
        this->ReceiveSendMsg("GetPhiOps", result);
    } else if (funcName == "IsDomInfoAvailable") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        bool ret = clientAPI.IsDomInfoAvailable();
        this->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)ret));
	} else if (funcName == "ConfirmValue") {
        /// Json格式
        /// {
        ///     "valId":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string valIdKey = "valId";
        uint64_t valId = atol(root[valIdKey].asString().c_str());
        mlir::Value v = clientAPI.GetValue(valId);
        Json::Value valueJson = ValueJsonSerialize(v);
        result = valueJson.toStyledString();
        this->ReceiveSendMsg("ValueResult", result);
    } else if (funcName == "BuildMemRef") {
        /// Json格式
        /// {
        ///     "baseId":"xxxx",
        ///     "offsetId":"xxxx",
        ///     "type":"xxxx",
        /// }
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        std::string baseIdKey = "baseId";
        std::string offsetIdKey = "offsetId";
        std::string typeKey = "type";
        uint64_t baseId = atol(root[baseIdKey].asString().c_str());
        uint64_t offsetId = atol(root[offsetIdKey].asString().c_str());
        Json::Value type = root[typeKey];
        PluginIR::PluginTypeBase pType = TypeJsonDeSerialize(type.toStyledString(), context);
        mlir::Value v = clientAPI.BuildMemRef(pType, baseId, offsetId);
        Json::Value valueJson = ValueJsonSerialize(v);
        result = valueJson.toStyledString();
        this->ReceiveSendMsg("ValueResult", result);
    } else {
        LOGW("function: %s not found!\n", funcName.c_str());
    }

    LOGD("IR function: %s\n", funcName.c_str());
    this->SetPluginAPIName("");
    this->SetUserFuncState(STATE_RETURN);
    this->ReceiveSendMsg(funcName, "done");
}

int PluginClient::GetInitInfo(string& serverPath, string& shaPath, int& timeout)
{
    Json::Value root;
    Json::Reader reader;
    std::ifstream ifs;
    string configFilePath;
    string serverDir;
    if (serverPath == "") {
        configFilePath = "/usr/local/bin/pin-gcc-client.json"; // server路径为空，配置文件默认路径
        LOGW("input serverPath is NULL, read default:%s\n", configFilePath.c_str());
    } else {
        int index = serverPath.find_last_of("/");
        serverDir = serverPath.substr(0, index);
        configFilePath = serverDir + "/pin-gcc-client.json"; // 配置文件和server在同一目录
    }

    ifs.open(configFilePath.c_str());
    if (!ifs) {
        shaPath = serverDir + "/libpin_user.sha256"; // sha256文件默认和server在同一目录
        LOGW("open %s fail! use default sha256file:%s\n", configFilePath.c_str(), shaPath.c_str());
        return -1;
    }
    reader.parse(ifs, root);
    ifs.close();

    if (serverPath == "") {
        if (!root["path"].isString()) {
            LOGE("path in config.json is not string\n");
            return 0;
        }
        serverPath = root["path"].asString();
        int index = serverPath.find_last_of("/");
        serverDir = serverPath.substr(0, index);
    }
    int timeoutJson = root["timeout"].asInt();
    if ((timeoutJson >= 50) && (timeoutJson <= 5000)) { // 不在50~5000ms范围内，使用默认值
        timeout = timeoutJson;
        LOGI("the timeout is:%d\n", timeout);
    } else {
        LOGW("timeout read from %s is:%d,should be 50~5000,use default:%d\n",
            configFilePath.c_str(), timeoutJson, timeout);
    }
    shaPath = root["sha256file"].asString();
    int ret = access(shaPath.c_str(), F_OK);
    if ((shaPath == "") || (ret != 0)) {
        shaPath = serverDir + "/libpin_user.sha256"; // sha256文件默认和server在同一目录
        LOGW("sha256 file not found,use default:%s\n", shaPath.c_str());
    }
    return 0;
}

void PluginClient::GetArg(struct plugin_name_args *pluginInfo, string& serverPath,
    string& arg, LogPriority& logLevel)
{
    Json::Value root;
    map<string, string> compileArgs;
    for (int i = 0; i < pluginInfo->argc; i++) {
        string key = pluginInfo->argv[i].key;
        if (key == "server_path") {
            serverPath = pluginInfo->argv[i].value;
        } else if (key == "log_level") {
            logLevel = (LogPriority)atoi(pluginInfo->argv[i].value);
            SetLogPriority(logLevel);
        } else {
            string value = pluginInfo->argv[i].value;
            compileArgs[key] = value;
            root[key] = value;
        }
    }
    arg = root.toStyledString();
    for (auto it = compileArgs.begin(); it != compileArgs.end(); it++) {
        CheckSafeCompileFlag(it->first, it->second);
    }
}

int PluginClient::CheckSHA256(const string& shaPath)
{
    if (shaPath == "") {
        LOGE("sha256file Path is NULL,check:%s\n", shaPath.c_str());
        return -1;
    }
    int index = shaPath.find_last_of("/");
    string dir = shaPath.substr(0, index);
    string filename = shaPath.substr(index+1, -1);

    string cmd = "cd " + dir + " && " + "sha256sum -c " + filename + " --quiet";
    int ret = system(cmd.c_str());
    return ret;
}

void PluginClient::CheckSafeCompileFlag(const string& argName, const string& param)
{
    vector<string> safeCompileFlags = {
        "-z noexecstack",
        "-fno-stack-protector",
        "-fstack-protector-all",
        "-D_FORTIFY_SOURCE",
        "-fPic",
        "-fPIE",
        "-fstack-protector-strong",
        "-fvisibility",
        "-ftrapv",
        "-fstack-check",
    };

    for (auto& v : safeCompileFlags) {
        if (param.find(v) != string::npos) {
            LOGW("%s:%s have safe compile parameter:%s !!!\n", argName.c_str(), param.c_str(), v.c_str());
        }
    }
}

bool PluginClient::DeletePortFromLockFile(unsigned short port)
{
    int portFileFd = open(g_portFilePath, O_RDWR);
    if (portFileFd == -1) {
        LOGE("%s open file %s fail\n", __func__, g_portFilePath);
        return false;
    }
    LOGI("delete port:%d\n", port);

    flock(portFileFd, LOCK_EX);
    string grpcPorts = "";
    ReadPortsFromLockFile(portFileFd, grpcPorts);

    string portStr = std::to_string(port) + "\n";
    string::size_type pos = grpcPorts.find(portStr);
    if (pos == string::npos) {
        close(portFileFd);
        return true;
    }
    grpcPorts = grpcPorts.erase(pos, portStr.size());

    ftruncate(portFileFd, 0);
    lseek(portFileFd, 0, SEEK_SET);
    write(portFileFd, grpcPorts.c_str(), grpcPorts.size());
    close(portFileFd);

    return true;
}

void PluginClient::ReceiveSendMsg(const string& attribute, const string& value)
{
    ClientContext context;
    auto stream = serviceStub->ReceiveSendMsg(&context);
    
    ClientMsg clientMsg;
    clientMsg.set_attribute(attribute);
    clientMsg.set_value(value);
    stream->Write(clientMsg);
    stream->WritesDone();
    TimerStart(timeout);
    if (g_grpcChannel->GetState(true) != GRPC_CHANNEL_READY) {
        LOGW("client pid:%d grpc channel not ready!\n", getpid());
        return;
    }
    ServerMsg serverMsg;
    while (stream->Read(&serverMsg)) {
        TimerStart(0);
        if (serverMsg.attribute() != "injectPoint") { // 日志不记录注册的函数名信息
            LOGD("rec from server:%s,%s\n", serverMsg.attribute().c_str(), serverMsg.value().c_str());
        }
        if ((serverMsg.attribute() == "start") && (serverMsg.value() == "ok")) {
            LOGI("server has been started!\n");
            if (!DeletePortFromLockFile(GetGrpcPort())) {
                LOGE("DeletePortFromLockFile fail\n");
            }
        } else if ((serverMsg.attribute() == "stop") && (serverMsg.value() == "ok")) {
            LOGI("server has been closed!\n");
            Status status = stream->Finish();
            if (!status.ok()) {
                LOGE("error code:%d,%s\n", status.error_code(), status.error_message().c_str());
                LOGE("RPC failed\n");
            }
            CloseLog();
        } else if ((serverMsg.attribute() == "userFunc") && (serverMsg.value() == "execution completed")) {
            SetUserFuncState(STATE_END); // server已接收到对应函数所需数据
        } else {
            ServerMsgProc(serverMsg.attribute(), serverMsg.value());
        }
    }
}

int PluginClient::AddRegisteredUserFunc(const string& value)
{
    int index = value.find_first_of(":");
    string point = value.substr(0, index);
    string name = value.substr(index + 1, -1);
    InjectPoint inject = (InjectPoint)atoi(point.c_str());
    if (inject >= HANDLE_MAX) {
        return -1;
    }
    
    registeredUserFunc[inject].push_back(name);
    return 0;
}

ManagerSetupData GetPassInfoData(const string& data)
{
    ManagerSetupData setupData;
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    
    setupData.refPassName = (RefPassName)root["refPassName"].asInt();
    setupData.passNum = root["passNum"].asInt();
    setupData.passPosition = (PassPosition)root["passPosition"].asInt();
    return setupData;
}

void PluginClient::ServerMsgProc(const string& attribute, const string& value)
{
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    if (attribute == "injectPoint") {
        if (value == "finished") {
            string pluginName = client->GetPluginName();
            InjectPoint inject;
            map<InjectPoint, vector<string>> userFuncs = client->GetRegisteredUserFunc();
            for (auto it = userFuncs.begin(); it != userFuncs.end(); it++) {
                inject = it->first;
                if (inject == HANDLE_MANAGER_SETUP) {
                    string data = it->second.front();
                    ManagerSetupData setupData = GetPassInfoData(data);
                    RegisterPassManagerSetup(inject, setupData, pluginName);
                } else {
                    RegisterPluginEvent(inject, pluginName); // 注册event
                }
            }
            client->SetInjectFlag(true);
        } else {
            client->AddRegisteredUserFunc(value);
        }
    } else {
        client->SetPluginAPIParam(value);
        client->SetPluginAPIName(attribute);
        client->SetUserFuncState(STATE_BEGIN);
    }
}

void TimeoutFunc(union sigval sig)
{
    LOGW("client pid:%d timeout!\n", getpid());
    g_plugin->SetUserFuncState(STATE_TIMEOUT);
}

void PluginClient::TimerStart(int interval)
{
    int msTons = 1000000; // ms转ns倍数
    int msTos = 1000; // s转ms倍数
    struct itimerspec time_value;
    time_value.it_value.tv_sec = (interval / msTos);
    time_value.it_value.tv_nsec = (interval % msTos) * msTons;
    time_value.it_interval.tv_sec = 0;
    time_value.it_interval.tv_nsec = 0;
    
    timer_settime(timerId, 0, &time_value, NULL);
}

bool PluginClient::TimerInit(void)
{
    struct sigevent evp;
    int sival = 124; // 传递整型参数，可以自定义
    memset(&evp, 0, sizeof(struct sigevent));
    evp.sigev_value.sival_ptr = timerId;
    evp.sigev_value.sival_int = sival;
    evp.sigev_notify = SIGEV_THREAD;
    evp.sigev_notify_function = TimeoutFunc;
    
    if (timer_create(CLOCK_REALTIME, &evp, &timerId) == -1) {
        LOGE("timer create fail\n");
        return false;
    }
    return true;
}

int PluginClient::OpenLockFile(const char *path)
{
    int portFileFd = -1;
    if (access(path, F_OK) == -1) {
        mode_t mask = umask(0);
        mode_t mode = 0666;
        portFileFd = open(path, O_CREAT | O_RDWR, mode);
        umask(mask);
    } else {
        portFileFd = open(path, O_RDWR);
    }

    if (portFileFd == -1) {
        LOGE("open file %s fail\n", path);
    }
    return portFileFd;
}

void PluginClient::ReadPortsFromLockFile(int fd, string& grpcPorts)
{
    int fileLen = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    char *buf = new char[fileLen + 1];
    read(fd, buf, fileLen);
    buf[fileLen] = '\0';
    grpcPorts = buf;
    delete[] buf;
}

unsigned short PluginClient::FindUnusedPort(void)
{
    unsigned short basePort = 40000; // grpc通信端口号从40000开始
    unsigned short foundPort = 0; // 可以使用的端口号
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr("0.0.0.0");

    int portFileFd = OpenLockFile(g_portFilePath);
    if (portFileFd == -1) {
        return 0;
    }

    flock(portFileFd, LOCK_EX);
    string grpcPorts = "";
    ReadPortsFromLockFile(portFileFd, grpcPorts);

    while (++basePort < UINT16_MAX) {
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        serverAddr.sin_port = htons(basePort);
        int ret = connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
        if (sock != -1) {
            close(sock);
        }
        if ((ret == -1) && (errno == ECONNREFUSED)) {
            string strPort = std::to_string(basePort) + "\n";
            if (grpcPorts.find(strPort) == grpcPorts.npos) {
                foundPort = basePort;
                LOGI("found port:%d\n", foundPort);
                lseek(portFileFd, 0, SEEK_END);
                write(portFileFd, strPort.c_str(), strPort.size());
                break;
            }
        }
    }

    if (basePort == UINT16_MAX) {
        ftruncate(portFileFd, 0);
        lseek(portFileFd, 0, SEEK_SET);
    }

    close(portFileFd); // 关闭文件fd会同时释放文件锁
    return foundPort;
}

int ServerStart(int timeout, const string& serverPath, pid_t& pid, string& port, const LogPriority logLevel)
{
    unsigned short portNum = PluginClient::FindUnusedPort();
    if (portNum == 0) {
        LOGE("cannot find port for grpc,port 40001-65535 all used!\n");
        return -1;
    }
    int ret = 0;
    port = std::to_string(portNum);
    pid = vfork();
    if (pid == 0) {
        LOGI("start plugin server!\n");
        string paramTimeout = std::to_string(timeout);
        if (execl(serverPath.c_str(), paramTimeout.c_str(), port.c_str(),
            std::to_string(logLevel).c_str(), NULL) == -1) {
            PluginClient::DeletePortFromLockFile(portNum);
            LOGE("server start fail! please check serverPath:%s\n", serverPath.c_str());
            ret = -1;
            _exit(0);
        }
    }
    int delay = 500000; // 500ms
    usleep(delay); // wait server start

    return ret;
}

int ClientStart(int timeout, const string& arg, const string& pluginName, const string& port)
{
    string serverPort = "localhost:" + port;
    g_grpcChannel = grpc::CreateChannel(serverPort, grpc::InsecureChannelCredentials());
    g_plugin = std::make_shared<PluginClient>(g_grpcChannel);
    g_plugin->SetInjectFlag(false);
    g_plugin->SetTimeout(timeout);
    g_plugin->SetUserFuncState(STATE_WAIT_BEGIN);

    if (!g_plugin->TimerInit()) {
        return 0;
    }
    unsigned short grpcPort = (unsigned short)atoi(port.c_str());
    g_plugin->SetGrpcPort(grpcPort);
    g_plugin->ReceiveSendMsg("start", arg);
    return 0;
}
} // namespace PinClient
