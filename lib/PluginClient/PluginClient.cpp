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
#include <semaphore.h>
#include "gccPlugin/gccPlugin.h"
#include "Dialect/PluginDialect.h"
#include "Dialect/PluginTypes.h"
#include "PluginAPI/PluginClientAPI.h"

namespace PinClient {
using namespace mlir::Plugin;
using namespace mlir;
static PluginClient *g_plugin;

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

PluginClient *PluginClient::GetInstance()
{
    if (g_plugin == nullptr) {
        g_plugin = new PluginClient();
    }
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

void PluginClient::GetGccData(const string& funcName, const string& param, string& key, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI pluginAPI(context);
    uint64_t gccDataAddr = (uint64_t)atol(param.c_str());
    if (gccDataAddr == 0) {
        LOGE("%s gcc_data address is NULL!\n", __func__);
        return;
    }
    if (funcName == "GetDeclSourceFile") {
        string sourceFile = pluginAPI.GetDeclSourceFile(gccDataAddr);
        json.StringSerialize(sourceFile, result);
        key = "StringResult";
    } else if (funcName == "GetDeclSourceLine") {
        int line = pluginAPI.GetDeclSourceLine(gccDataAddr);
        json.IntegerSerialize(line, result);
        key = "IntegerResult";
    } else if (funcName == "GetDeclSourceColumn") {
        int column = pluginAPI.GetDeclSourceColumn(gccDataAddr);
        json.IntegerSerialize(column, result);
        key = "IntegerResult";
    } else if (funcName == "VariableName") {
        string variableName = pluginAPI.VariableName(gccDataAddr);
        json.StringSerialize(variableName, result);
        key = "StringResult";
    } else if (funcName == "FuncName") {
        string funcName = pluginAPI.FuncName(gccDataAddr);
        json.StringSerialize(funcName, result);
        key = "StringResult";
    } else {
        LOGW("function: %s not found!\n", funcName.c_str());
    }
}

// CGnode ============

void GetCGnodeIDsResult(PluginClient *client, Json::Value& root, string& result)
{
    // Load our Dialect in this MLIR Context.
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<uint64_t> ids = clientAPI.GetCGnodeIDs();
    PluginJson json = client->GetJson();
    json.IDsJsonSerialize(ids, result);
    client->ReceiveSendMsg("IdsResult", result);
}

void GetCGnodeOpByIdResult(PluginClient *client, Json::Value& root, string& result)
{
    // Load our Dialect in this MLIR Context.
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string funcIdKey = "id";
    uint64_t cgnodeID = atol(root[funcIdKey].asString().c_str());
    CGnodeOp cgnodeOp = clientAPI.GetCGnodeOpById(cgnodeID);
    PluginJson json = client->GetJson();
    json.CGnodeOpJsonSerialize(cgnodeOp, result);
    client->ReceiveSendMsg("CGnodeOpResult", result);
}

void IsRealSymbolOfCGnodeResult(PluginClient *client, Json::Value& root, string& result)
{
    // Load our Dialect in this MLIR Context.
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string funcIdKey = "id";
    uint64_t cgnodeID = atol(root[funcIdKey].asString().c_str());
    bool realsymbol = clientAPI.IsRealSymbolOfCGnode(cgnodeID);
    client->ReceiveSendMsg("BoolResult", std::to_string(realsymbol));
}

// ===================

void GetAllFuncResult(PluginClient *client, Json::Value& root, string& result)
{
    // Load our Dialect in this MLIR Context.
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<FunctionOp> allFuncOps = clientAPI.GetAllFunc();
    PluginJson json = client->GetJson();
    json.FunctionOpJsonSerialize(allFuncOps, result);
    client->ReceiveSendMsg("FuncOpResult", result);
}

void GetFunctionIDsResult(PluginClient *client, Json::Value& root, string& result)
{
    // Load our Dialect in this MLIR Context.
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<uint64_t> ids = clientAPI.GetFunctions();
    PluginJson json = client->GetJson();
    json.IDsJsonSerialize(ids, result);
    client->ReceiveSendMsg("IdsResult", result);
}

void GetFunctionOpByIdResult(PluginClient *client, Json::Value& root, string& result)
{
    // Load our Dialect in this MLIR Context.
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string funcIdKey = "id";
    uint64_t funcID = atol(root[funcIdKey].asString().c_str());
    vector<FunctionOp> allFuncOps;
    allFuncOps.push_back(clientAPI.GetFunctionOpById(funcID));
    PluginJson json = client->GetJson();
    json.FunctionOpJsonSerialize(allFuncOps, result);
    client->ReceiveSendMsg("FuncOpResult", result);
}

void GetLocalDeclsResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.LocalDeclsJsonSerialize(decls, result);
    client->ReceiveSendMsg("LocalDeclOpResult", result);
}

void GetFuncDeclsResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "funcId":"xxxx"
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string funcIdKey = "funcId";
    uint64_t funcID = atol(root[funcIdKey].asString().c_str());
    vector<DeclBaseOp> decls = clientAPI.GetFuncDecls(funcID);
    PluginJson json = client->GetJson();
    json.FunctionDeclsJsonSerialize(decls, result);
    client->ReceiveSendMsg("FuncDeclsOpResult", result);
}

void GetMakeNodeResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "defCode":"xxxx"
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string codeKey = "defCode";
    IDefineCode code = IDefineCode(atoi((root[codeKey].asString().c_str())));
    mlir::Value v = clientAPI.MakeNode(code);
    PluginJson json = client->GetJson();
    result = json.ValueJsonSerialize(v).toStyledString();
    client->ReceiveSendMsg("MakeNodeResult", result);
}

void GetFieldsResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "declId":"xxxx"
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string declIdKey = "declId";
    uint64_t declID = atol(root[declIdKey].asString().c_str());
    vector<FieldDeclOp> decls = clientAPI.GetFields(declID);
    PluginJson json = client->GetJson();
    json.FiledOpsJsonSerialize(decls, result);
    client->ReceiveSendMsg("GetFieldsOpResult", result);
}

void GetBuildDeclResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "defCode":"xxxx",
    ///     "name":"xxxx",
    ///     "type":"xxxx"
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string defCode = "defCode";
    std::string name = "name";
    std::string type = "type";
    IDefineCode code = IDefineCode(atoi((root[defCode].asString().c_str())));
    string tname = root[name].asString();
    PluginJson json = client->GetJson();
    PluginIR::PluginTypeBase t = json.TypeJsonDeSerialize(root[type].toStyledString(), context);
    DeclBaseOp decl = clientAPI.BuildDecl(code, tname, t);

    result = json.DeclBaseOpJsonSerialize(decl).toStyledString();
    client->ReceiveSendMsg("DeclOpResult", result);
}

void GetDeclTypeResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string declIdKey = "declId";
    uint64_t declID = atol(root[declIdKey].asString().c_str());
    PluginIR::PluginTypeBase retType = clientAPI.GetDeclType(declID);
    PluginJson json = client->GetJson();
    result = json.TypeJsonSerialize(retType).toStyledString();
    client->ReceiveSendMsg("PluginTypeResult", result);
}

void SetDeclNameResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetDeclName(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetSourceLocationResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetSourceLocation(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetDeclAlignResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetDeclAlign(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetUserAlignResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetUserAlign(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetTypeFieldsResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "declId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string declIdkey = "declId";
    uint64_t declId = atol(root[declIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetTypeFields(declId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void LayoutTypeResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "declId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string declIdKey = "declId";
    uint64_t declId = atol(root[declIdKey].asString().c_str());
    clientAPI.LayoutType(declId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void LayoutDeclResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "declId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string declIdKey = "declId";
    uint64_t declId = atol(root[declIdKey].asString().c_str());
    clientAPI.LayoutDecl(declId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}


void SetDeclChainResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetDeclChain(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void GetDeclTypeSizeResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "declId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string declIdkey = "declId";
    uint64_t declId = atol(root[declIdkey].asString().c_str());
    unsigned size = clientAPI.GetDeclTypeSize(declId);
    PluginJson json = client->GetJson();
    json.IntegerSerialize(size, result);
    client->ReceiveSendMsg("IntegerResult", result);
}

void SetAddressableResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetAddressable(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetNonAddressablepResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetNonAddressablep(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetVolatileResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetVolatile(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetDeclContextResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string declIdKey = "declId";
    uint64_t declId = atol(root[declIdKey].asString().c_str());
    clientAPI.SetDeclContext(newfieldId, declId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetDeclTypeResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "newfieldId":"xxxx",
    ///     "fieldId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string newfieldIdkey = "newfieldId";
    uint64_t newfieldId = atol(root[newfieldIdkey].asString().c_str());
    std::string fieldIdKey = "fieldId";
    uint64_t fieldId = atol(root[fieldIdKey].asString().c_str());
    clientAPI.SetDeclType(newfieldId, fieldId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void GetLoopsFromFuncResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.LoopOpsJsonSerialize(irLoops, result);
    client->ReceiveSendMsg("LoopOpsResult", result);
}

void GetLoopByIdResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.LoopOpJsonSerialize(irLoop, result);
    client->ReceiveSendMsg("LoopOpResult", result);
}

void IsBlockInsideResult(PluginClient *client, Json::Value& root, string& result)
{
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
    client->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)res));
}

void AllocateNewLoopResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t newLoopId = clientAPI.AllocateNewLoop();
    LoopOp newLoop = clientAPI.GetLoopById(newLoopId);
    PluginJson json = client->GetJson();
    json.LoopOpJsonSerialize(newLoop, result);
    client->ReceiveSendMsg("LoopOpResult", result);
}

void RedirectFallthroughTargetResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "src":"xxxx",
    ///     "dest":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string srcKey = "src";
    uint64_t src = atol(root[srcKey].asString().c_str());
    std::string destKey = "dest";
    uint64_t dest = atol(root[destKey].asString().c_str());
    clientAPI.RedirectFallthroughTarget(src, dest);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void DeleteLoopResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void AddBlockToLoopResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "loopId":"xxxx",
    ///     "blockId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string loopIdKey = "loopId";
    uint64_t loopId = atol(root[loopIdKey].asString().c_str());
    std::string blockIdKey = "blockId";
    uint64_t blockId = atol(root[blockIdKey].asString().c_str());
    clientAPI.AddBlockToLoop(blockId, loopId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void AddLoopResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.LoopOpJsonSerialize(irLoop, result);
    client->ReceiveSendMsg("LoopOpResult", result);
}

void GetBlocksInLoopResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "loopId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string loopIdKey = "loopId";
    uint64_t loopId = atol(root[loopIdKey].asString().c_str());
    vector<uint64_t> blockIDs = clientAPI.GetBlocksInLoop(loopId);
    PluginJson json = client->GetJson();
    json.IDsJsonSerialize(blockIDs, result);
    client->ReceiveSendMsg("IdsResult", result);
}

void GetHeaderResult(PluginClient *client, Json::Value& root, string& result)
{
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
    client->ReceiveSendMsg("IdResult", std::to_string(blockId));
}

void GetLatchResult(PluginClient *client, Json::Value& root, string& result)
{
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
    client->ReceiveSendMsg("IdResult", std::to_string(blockId));
}

void SetHeaderResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "loopId":"xxxx",
    ///     "blockId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string loopIdKey = "loopId";
    uint64_t loopId = atol(root[loopIdKey].asString().c_str());
    std::string blockIdKey = "blockId";
    uint64_t blockId = atol(root[blockIdKey].asString().c_str());
    clientAPI.SetHeader(loopId, blockId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetLatchResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "loopId":"xxxx",
    ///     "blockId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string loopIdKey = "loopId";
    uint64_t loopId = atol(root[loopIdKey].asString().c_str());
    std::string blockIdKey = "blockId";
    uint64_t blockId = atol(root[blockIdKey].asString().c_str());
    clientAPI.SetLatch(loopId, blockId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void GetLoopExitsResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "loopId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string loopIdKey = "loopId";
    uint64_t loopId = atol(root[loopIdKey].asString().c_str());
    vector<std::pair<uint64_t, uint64_t> > edges = clientAPI.GetLoopExits(loopId);
    PluginJson json = client->GetJson();
    json.EdgesJsonSerialize(edges, result);
    client->ReceiveSendMsg("EdgesResult", result);
}

void GetLoopSingleExitResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "loopId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string loopIdKey = "loopId";
    uint64_t loopId = atol(root[loopIdKey].asString().c_str());
    std::pair<uint64_t, uint64_t> edge = clientAPI.GetLoopSingleExit(loopId);
    PluginJson json = client->GetJson();
    json.EdgeJsonSerialize(edge, result);
    client->ReceiveSendMsg("EdgeResult", result);
}

void GetBlockLoopFatherResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.LoopOpJsonSerialize(loopFather, result);
    client->ReceiveSendMsg("LoopOpResult", result);
}

void FindCommonLoopResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t loopId_1 = atol(root["loopId_1"].asString().c_str());
    uint64_t loopId_2 = atol(root["loopId_2"].asString().c_str());
    LoopOp commonLoop = clientAPI.FindCommonLoop(loopId_1, loopId_2);
    PluginJson json = client->GetJson();
    json.LoopOpJsonSerialize(commonLoop, result);
    client->ReceiveSendMsg("LoopOpResult", result);
}

void CreateBlockResult(PluginClient *client, Json::Value& root, string& result)
{
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
    client->ReceiveSendMsg("IdResult", std::to_string(newBBAddr));
}

void DeleteBlockResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "bbaddr":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string funcKey = "funcaddr";
    std::string BlockIdKey = "bbaddr";
    uint64_t bbaddr = atol(root[BlockIdKey].asString().c_str());
    uint64_t funcaddr = atol(root[funcKey].asString().c_str());
    clientAPI.DeleteBlock(funcaddr, bbaddr);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void SetImmediateDominatorResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void GetImmediateDominatorResult(PluginClient *client, Json::Value& root, string& result)
{
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
    client->ReceiveSendMsg("IdResult", std::to_string(ret));
}

void RecomputeDominatorResult(PluginClient *client, Json::Value& root, string& result)
{
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
    client->ReceiveSendMsg("IdResult", std::to_string(ret));
}

void GetPhiOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t id = atol(root[std::to_string(0)].asString().c_str());
    PhiOp op = clientAPI.GetPhiOp(id);
    PluginJson json = client->GetJson();
    Json::Value phiOpResult = json.PhiOpJsonSerialize(op);
    client->ReceiveSendMsg("OpResult", phiOpResult.toStyledString());
}

void GetCallOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t id = atol(root[std::to_string(0)].asString().c_str());
    CallOp op = clientAPI.GetCallOp(id);
    PluginJson json = client->GetJson();
    Json::Value opResult = json.CallOpJsonSerialize(op);
    client->ReceiveSendMsg("OpResult", opResult.toStyledString());
}

void SetLhsInCallOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t callId = atol(root["callId"].asString().c_str());
    uint64_t lhsId = atol(root["lhsId"].asString().c_str());
    bool ret = clientAPI.SetLhsInCallOp(callId, lhsId);
    client->ReceiveSendMsg("BoolResult", std::to_string(ret));
}

void AddArgInPhiOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t phiId = atol(root["phiId"].asString().c_str());
    uint64_t argId = atol(root["argId"].asString().c_str());
    uint64_t predId = atol(root["predId"].asString().c_str());
    uint64_t succId = atol(root["succId"].asString().c_str());
    uint32_t ret = clientAPI.AddArgInPhiOp(phiId, argId, predId, succId);
    client->ReceiveSendMsg("IdResult", std::to_string(ret));
}

void CreateCallOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t blockId = atol(root["blockId"].asString().c_str());
    uint64_t funcId = atol(root["funcId"].asString().c_str());
    vector<uint64_t> argIds;
    Json::Value argIdsJson = root["argIds"];
    Json::Value::Members member = argIdsJson.getMemberNames();
    for (Json::Value::Members::iterator opIter = member.begin(); opIter != member.end(); opIter++) {
        string key = *opIter;
        uint64_t id = atol(argIdsJson[key.c_str()].asString().c_str());
        argIds.push_back(id);
    }
    uint64_t ret = clientAPI.CreateCallOp(blockId, funcId, argIds);
    client->ReceiveSendMsg("IdResult", std::to_string(ret));
}

void CreateAssignOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t blockId = atol(root["blockId"].asString().c_str());
    int condCode = atol(root["exprCode"].asString().c_str());
    vector<uint64_t> argIds;
    Json::Value argIdsJson = root["argIds"];
    Json::Value::Members member = argIdsJson.getMemberNames();
    for (Json::Value::Members::iterator opIter = member.begin(); opIter != member.end(); opIter++) {
        string key = *opIter;
        uint64_t id = atol(argIdsJson[key.c_str()].asString().c_str());
        argIds.push_back(id);
    }
    uint64_t ret = clientAPI.CreateAssignOp(blockId, IExprCode(condCode), argIds);
    client->ReceiveSendMsg("IdResult", std::to_string(ret));
}

void CreateCondOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t blockId = atol(root["blockId"].asString().c_str());
    int condCode = atol(root["condCode"].asString().c_str());
    uint64_t lhsId = atol(root["lhsId"].asString().c_str());
    uint64_t rhsId = atol(root["rhsId"].asString().c_str());
    uint64_t tbaddr = atol(root["tbaddr"].asString().c_str());
    uint64_t fbaddr = atol(root["fbaddr"].asString().c_str());
    uint64_t ret = clientAPI.CreateCondOp(blockId, IComparisonCode(condCode), lhsId, rhsId, tbaddr, fbaddr);
    client->ReceiveSendMsg("IdResult", std::to_string(ret));
}

void CreateFallthroughOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t address = atol(root["address"].asString().c_str());
    uint64_t destaddr = atol(root["destaddr"].asString().c_str());
    clientAPI.CreateFallthroughOp(address, destaddr);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void GetResultFromPhiResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t id = atol(root["id"].asString().c_str());
    mlir::Value ret = clientAPI.GetResultFromPhi(id);
    PluginJson json = client->GetJson();
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(ret).toStyledString());
}

void CreatePhiOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t argId = atol(root["argId"].asString().c_str());
    uint64_t blockId = atol(root["blockId"].asString().c_str());
    PhiOp op = clientAPI.CreatePhiOp(argId, blockId);
    PluginJson json = client->GetJson();
    Json::Value opResult = json.PhiOpJsonSerialize(op);
    client->ReceiveSendMsg("OpResult", opResult.toStyledString());
}

void CreateConstOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    PluginJson json = client->GetJson();
    PluginIR::PluginTypeBase type = json.TypeJsonDeSerialize(root.toStyledString(), context);
    uint64_t value = atol(root["value"].asString().c_str());
    mlir::OpBuilder opBuilder = mlir::OpBuilder(&context);
    mlir::Attribute attr = opBuilder.getI64IntegerAttr(value);
    mlir::Value ret = clientAPI.CreateConstOp(attr, type);
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(ret).toStyledString());
}

void UpdateSSAResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.UpdateSSA();
    client->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)ret));
}

void GetAllPhiOpInsideBlockResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t bb = atol(root["bbAddr"].asString().c_str());
    vector<PhiOp> phiOps = clientAPI.GetPhiOpsInsideBlock(bb);
    PluginJson json = client->GetJson();
    json.GetPhiOpsJsonSerialize(phiOps, result);
    client->ReceiveSendMsg("GetPhiOps", result);
}

void GetAllOpsInsideBlockResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t bb = atol(root["bbAddr"].asString().c_str());
    vector<uint64_t> opsId = clientAPI.GetOpsInsideBlock(bb);
    PluginJson json = client->GetJson();
    json.IDsJsonSerialize(opsId, result);
    client->ReceiveSendMsg("IdsResult", result);
}

void IsDomInfoAvailableResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.IsDomInfoAvailable();
    client->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)ret));
}

void GetCurrentDefFromSSAResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t varId = atol(root["varId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    mlir::Value ret = clientAPI.GetCurrentDefFromSSA(varId);
    PluginJson json = client->GetJson();
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(ret).toStyledString());
}

void SetCurrentDefInSSAResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t varId = atol(root["varId"].asString().c_str());
    uint64_t defId = atol(root["defId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.SetCurrentDefInSSA(varId, defId);
    client->ReceiveSendMsg("BoolResult", std::to_string((uint64_t)ret));
}

void CopySSAOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t id = atol(root["id"].asString().c_str());
    mlir::Value ret = clientAPI.CopySSAOp(id);
    PluginJson json = client->GetJson();
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(ret).toStyledString());
}

void CreateSSAOpResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    PluginJson json = client->GetJson();
    PluginIR::PluginTypeBase type = json.TypeJsonDeSerialize(root.toStyledString(), context);
    mlir::Value ret = clientAPI.CreateSSAOp(type);
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(ret).toStyledString());
}

void CreateNewDefResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t opId = atol(root["opId"].asString().c_str());
    uint64_t valueId = atol(root["valueId"].asString().c_str());
    uint64_t defId = atol(root["defId"].asString().c_str());
    mlir::Value ret = clientAPI.CreateNewDef(valueId, opId, defId);
    PluginJson json = client->GetJson();
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(ret).toStyledString());
}

void CalDominanceInfoResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string dirIdKey = "dir";
    uint64_t dir = atol(root[dirIdKey].asString().c_str());
    uint64_t funcId = atol(root["funcId"].asString().c_str());
    clientAPI.CalDominanceInfo(dir, funcId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void GetImmUseStmtsResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t varId = atol(root["varId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<uint64_t> opsId = clientAPI.GetImmUseStmts(varId);
    PluginJson json = client->GetJson();
    json.IDsJsonSerialize(opsId, result);
    client->ReceiveSendMsg("IdsResult", result);
}

void GetGimpleVuseResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t opId = atol(root["opId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    mlir::Value vuse = clientAPI.GetGimpleVuse(opId);
    PluginJson json = client->GetJson();
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(vuse).toStyledString());
}

void GetGimpleVdefResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t opId = atol(root["opId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    mlir::Value vdef = clientAPI.GetGimpleVdef(opId);
    PluginJson json = client->GetJson();
    client->ReceiveSendMsg("ValueResult", json.ValueJsonSerialize(vdef).toStyledString());
}

void GetSsaUseOperandResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t opId = atol(root["opId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<mlir::Value> ret = clientAPI.GetSsaUseOperand(opId);
    PluginJson json = client->GetJson();
    json.ValuesJsonSerialize(ret, result);
    client->ReceiveSendMsg("ValuesResult", result);
}

void GetSsaDefOperandResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t opId = atol(root["opId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<mlir::Value> ret = clientAPI.GetSsaDefOperand(opId);
    PluginJson json = client->GetJson();
    json.ValuesJsonSerialize(ret, result);
    client->ReceiveSendMsg("ValuesResult", result);
}

void GetPhiOrStmtUseResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t opId = atol(root["opId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<mlir::Value> ret = clientAPI.GetPhiOrStmtUse(opId);
    PluginJson json = client->GetJson();
    json.ValuesJsonSerialize(ret, result);
    client->ReceiveSendMsg("ValuesResult", result);
}

void GetPhiOrStmtDefResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t opId = atol(root["opId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    vector<mlir::Value> ret = clientAPI.GetPhiOrStmtDef(opId);
    PluginJson json = client->GetJson();
    json.ValuesJsonSerialize(ret, result);
    client->ReceiveSendMsg("ValuesResult", result);
}

void RefsMayAliasResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t id1 = atol(root["id1"].asString().c_str());
    uint64_t id2 = atol(root["id2"].asString().c_str());
    uint64_t flag = atol(root["flag"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.RefsMayAlias(id1, id2, flag);
    client->ReceiveSendMsg("BoolResult", std::to_string(ret));
}

void PTIncludesDeclResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t ptrId = atol(root["ptrId"].asString().c_str());
    uint64_t declId = atol(root["declId"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.PTIncludesDecl(ptrId, declId);
    client->ReceiveSendMsg("BoolResult", std::to_string(ret));
}

void PTsIntersectResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t ptrId_1 = atol(root["ptrId_1"].asString().c_str());
    uint64_t ptrId_2 = atol(root["ptrId_2"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.PTsIntersect(ptrId_1, ptrId_2);
    client->ReceiveSendMsg("BoolResult", std::to_string(ret));
}

void RemoveEdgeResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "src":"xxxx",
    ///     "dest":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string srcKey = "src";
    uint64_t src = atol(root[srcKey].asString().c_str());
    std::string destKey = "dest";
    uint64_t dest = atol(root[destKey].asString().c_str());
    clientAPI.RemoveEdge(src, dest);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void ConfirmValueResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    Json::Value valueJson = json.ValueJsonSerialize(v);
    result = valueJson.toStyledString();
    client->ReceiveSendMsg("ValueResult", result);
}

void BuildMemRefResult(PluginClient *client, Json::Value& root, string& result)
{
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
    PluginJson json = client->GetJson();
    PluginIR::PluginTypeBase pType = json.TypeJsonDeSerialize(type.toStyledString(), context);
    mlir::Value v = clientAPI.BuildMemRef(pType, baseId, offsetId);
    Json::Value valueJson = json.ValueJsonSerialize(v);
    result = valueJson.toStyledString();
    client->ReceiveSendMsg("ValueResult", result);
}

void IsVirtualOperandResult(PluginClient *client, Json::Value& root, string& result)
{
     mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    uint64_t id = atol(root["id"].asString().c_str());
    PluginAPI::PluginClientAPI clientAPI(context);
    bool ret = clientAPI.IsVirtualOperand(id);
    client->ReceiveSendMsg("BoolResult", std::to_string(ret));
}

void DebugValueResult(PluginClient *client, Json::Value& root, string& result)
{
    /// Json格式
    /// {
    ///     "valId":"xxxx",
    /// }
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string valIdKey = "valId";
    uint64_t valId = atol(root[valIdKey].asString().c_str());
    clientAPI.DebugValue(valId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("ValueResult", result);
}

void DebugOperationResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    std::string opIdKey = "opId";
    uint64_t opId = atol(root[opIdKey].asString().c_str());
    clientAPI.DebugOperation(opId);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void DebugBlockResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    uint64_t bb = atol(root["bbAddr"].asString().c_str());
    clientAPI.DebugBlock(bb);
    PluginJson json = client->GetJson();
    json.NopJsonSerialize(result);
    client->ReceiveSendMsg("VoidResult", result);
}

void IsLtoOptimizeResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    bool lto = clientAPI.IsLtoOptimize();
    client->ReceiveSendMsg("BoolResult", std::to_string(lto));
}

void IsWholeProgramResult(PluginClient *client, Json::Value& root, string& result)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<PluginDialect>();
    PluginAPI::PluginClientAPI clientAPI(context);
    bool wholePR = clientAPI.IsWholeProgram();
    client->ReceiveSendMsg("BoolResult", std::to_string(wholePR));
}

typedef std::function<void(PluginClient*, Json::Value&, string&)> GetResultFunc;
std::map<string, GetResultFunc> g_getResultFunc = {
    {"GetCGnodeIDs", GetCGnodeIDsResult},
    {"GetCGnodeOpById", GetCGnodeOpByIdResult},
    {"IsRealSymbolOfCGnode", IsRealSymbolOfCGnodeResult},
    {"GetAllFunc", GetAllFuncResult},
    {"GetFunctionIDs", GetFunctionIDsResult},
    {"GetFunctionOpById", GetFunctionOpByIdResult},
    {"GetLocalDecls", GetLocalDeclsResult},
    {"GetFuncDecls", GetFuncDeclsResult},
    {"GetFields", GetFieldsResult},
    {"BuildDecl", GetBuildDeclResult},
    {"GetDeclType", GetDeclTypeResult},
    {"MakeNode", GetMakeNodeResult},
    {"SetDeclName", SetDeclNameResult},
    {"SetDeclType", SetDeclTypeResult},
    {"SetSourceLocation", SetSourceLocationResult},
    {"SetDeclAlign", SetDeclAlignResult},
    {"SetUserAlign", SetUserAlignResult},
    {"SetTypeFields", SetTypeFieldsResult},
    {"LayoutType", LayoutTypeResult},
    {"LayoutDecl", LayoutDeclResult},
    {"SetAddressable", SetAddressableResult},
    {"GetDeclTypeSize", GetDeclTypeSizeResult},
    {"SetDeclChain", SetDeclChainResult},
    {"SetNonAddressablep", SetNonAddressablepResult},
    {"SetVolatile", SetVolatileResult},
    {"SetDeclContext", SetDeclContextResult},
    {"GetLoopsFromFunc", GetLoopsFromFuncResult},
    {"GetLoopById", GetLoopByIdResult},
    {"IsBlockInside", IsBlockInsideResult},
    {"AllocateNewLoop", AllocateNewLoopResult},
    {"RedirectFallthroughTarget", RedirectFallthroughTargetResult},
    {"DeleteLoop", DeleteLoopResult},
    {"AddBlockToLoop", AddBlockToLoopResult},
    {"AddLoop", AddLoopResult},
    {"GetBlocksInLoop", GetBlocksInLoopResult},
    {"GetHeader", GetHeaderResult},
    {"GetLatch", GetLatchResult},
    {"SetHeader", SetHeaderResult},
    {"SetLatch", SetLatchResult},
    {"GetLoopExits", GetLoopExitsResult},
    {"GetLoopSingleExit", GetLoopSingleExitResult},
    {"GetBlockLoopFather", GetBlockLoopFatherResult},
    {"FindCommonLoop", FindCommonLoopResult},
    {"CreateBlock", CreateBlockResult},
    {"DeleteBlock", DeleteBlockResult},
    {"SetImmediateDominator", SetImmediateDominatorResult},
    {"GetImmediateDominator", GetImmediateDominatorResult},
    {"RecomputeDominator", RecomputeDominatorResult},
    {"GetPhiOp", GetPhiOpResult},
    {"GetCallOp", GetCallOpResult},
    {"SetLhsInCallOp", SetLhsInCallOpResult},
    {"AddArgInPhiOp", AddArgInPhiOpResult},
    {"CreateCallOp", CreateCallOpResult},
    {"CreateAssignOp", CreateAssignOpResult},
    {"CreateCondOp", CreateCondOpResult},
    {"CreateFallthroughOp", CreateFallthroughOpResult},
    {"GetResultFromPhi", GetResultFromPhiResult},
    {"CreatePhiOp", CreatePhiOpResult},
    {"CreateConstOp", CreateConstOpResult},
    {"UpdateSSA", UpdateSSAResult},
    {"GetAllPhiOpInsideBlock", GetAllPhiOpInsideBlockResult},
    {"GetAllOpsInsideBlock", GetAllOpsInsideBlockResult},
    {"IsDomInfoAvailable", IsDomInfoAvailableResult},
    {"GetCurrentDefFromSSA", GetCurrentDefFromSSAResult},
    {"SetCurrentDefInSSA", SetCurrentDefInSSAResult},
    {"CopySSAOp", CopySSAOpResult},
    {"CreateSSAOp", CreateSSAOpResult},
    {"CreateNewDef", CreateNewDefResult},
    {"RemoveEdge", RemoveEdgeResult},
    {"ConfirmValue", ConfirmValueResult},
    {"BuildMemRef", BuildMemRefResult},
    {"IsVirtualOperand", IsVirtualOperandResult},
    {"DebugValue", DebugValueResult},
    {"DebugOperation", DebugOperationResult},
    {"DebugBlock", DebugBlockResult},
    {"IsLtoOptimize", IsLtoOptimizeResult},
    {"IsWholeProgram", IsWholeProgramResult},
    {"CalDominanceInfo", CalDominanceInfoResult},
    {"GetImmUseStmts", GetImmUseStmtsResult},
    {"GetGimpleVuse", GetGimpleVuseResult},
    {"GetGimpleVdef", GetGimpleVdefResult},
    {"GetSsaUseOperand", GetSsaUseOperandResult},
    {"GetSsaDefOperand", GetSsaDefOperandResult},
    {"GetPhiOrStmtUse", GetPhiOrStmtUseResult},
    {"GetPhiOrStmtDef", GetPhiOrStmtDefResult},
    {"RefsMayAlias", RefsMayAliasResult},
    {"PTIncludesDecl", PTIncludesDeclResult},
    {"PTsIntersect", PTsIntersectResult}
};

void PluginClient::GetIRTransResult(void *gccData, const string& funcName, const string& param)
{
    string result;
    Json::Value root;
    Json::Reader reader;
    reader.parse(param, root);
    LOGD("%s func:%s,param:%s\n", __func__, funcName.c_str(), param.c_str());

    if (funcName == "GetInjectDataAddress") {
        int64_t ptrAddress = (int64_t)gccData;
        json.IntegerSerialize(ptrAddress, result);
        this->ReceiveSendMsg("IntegerResult", result);
    } else if (funcName == "GetIncludeFile") {
        if (gccData != nullptr) {
            string includeFile = (char *)gccData;
            json.StringSerialize(includeFile, result);
            this->ReceiveSendMsg("StringResult", result);
        } else {
            LOGE("%s gcc_data address is NULL!\n", __func__);
        }
    } else {
        auto it = g_getResultFunc.find(funcName);
        if (it != g_getResultFunc.end()) {
            it->second(this, root, result);
        } else {
            string key = "";
            GetGccData(funcName, param, key, result);
            if (key != "") {
                this->ReceiveSendMsg(key, result);
            }
        }
    }

    LOGD("IR function: %s\n", funcName.c_str());
    this->SetPluginAPIName("");
    this->SetUserFuncState(STATE_RETURN);
    this->ReceiveSendMsg(funcName, "done");
}

// attribute:value说明
// start:ok 启动成功
// stop:ok  关闭成功
// userFunc:execution completed 函数执行完毕，执行下一个函数
// injectPoit:xxxx 注册点，xxxx是server传递过来的唯一值，gcc回调时，将xxxx返回server解析，执行对应的函数
// xxxx:yyyy 此类都认为是api函数，xxxx为函数名，yyyy为形参
void PluginClient::ReceiveSendMsg(const string& attribute, const string& value)
{
    ClientContext context;
    auto stream = serviceStub->ReceiveSendMsg(&context);
    
    ClientMsg clientMsg;
    clientMsg.set_attribute(attribute);
    clientMsg.set_value(value);
    stream->Write(clientMsg);
    stream->WritesDone();
    TimerStart(input.GetTimeout());

    if (grpcChannel->GetState(true) != GRPC_CHANNEL_READY) {
        LOGW("client pid:%d grpc channel not ready!\n", getpid());
        return;
    }
    ServerMsg serverMsg;
    while (stream->Read(&serverMsg)) {
        TimerStart(0); // 定时值0，关闭定时器
        if (serverMsg.attribute() != grpckey[INJECT]) { // 日志不记录注册的函数名信息
            LOGD("rec from server:%s,%s\n", serverMsg.attribute().c_str(), serverMsg.value().c_str());
        }
        if ((serverMsg.attribute() == grpckey[START]) && (serverMsg.value() == grpcValue[START_VALUE])) {
            LOGI("server has been started!\n");
            DeleteGrpcPort();
        } else if ((serverMsg.attribute() == grpckey[STOP]) && (serverMsg.value() == grpcValue[STOP_VALUE])) {
            LOGI("server has been closed!\n");
            Status status = stream->Finish();
            if (!status.ok()) {
                LOGE("RPC failed,error code:%d,%s\n", status.error_code(), status.error_message().c_str());
            }
            CloseLog();
        } else if ((serverMsg.attribute() == grpckey[USERFUNC]) && (serverMsg.value() == grpcValue[USERFUNC_VALUE])) {
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
        LOGE("AddRegisteredUserFunc %s err!\n", value.c_str());
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
    string jsonData = data.substr(data.find_first_of(":") + 1, -1);
    reader.parse(jsonData, root);
    
    if (root[passValue[PASS_NAME]].isInt()) {
        setupData.refPassName = (RefPassName)root[passValue[PASS_NAME]].asInt();
    }
    if (root[passValue[PASS_NUM]].isInt()) {
        setupData.passNum = root[passValue[PASS_NUM]].asInt();
    }
    if (root[passValue[PASS_POSITION]].isInt()) {
        setupData.passPosition = (PassPosition)root[passValue[PASS_POSITION]].asInt();
    }

    return setupData;
}

// attribute不为"injectPoint"时,attribute为"funcName",value为函数参数
// attribute为"injectPoint"时, value为"inject:funcName+函数指针地址"
void PluginClient::ServerMsgProc(const string& attribute, const string& value)
{
    if (attribute != grpckey[INJECT]) {
        SetPluginAPIParam(value);
        SetPluginAPIName(attribute);
        SetUserFuncState(STATE_BEGIN);
        return;
    }

    if (value == grpcValue[FINISH_VALUE]) {
        string pluginName = GetPluginName();
        InjectPoint inject;
        map<InjectPoint, vector<string>> userFuncs = GetRegisteredUserFunc();
        for (auto it = userFuncs.begin(); it != userFuncs.end(); it++) {
            inject = it->first; // first为注册点
            if (inject == HANDLE_MANAGER_SETUP) {
                for (unsigned int i = 0; i < it->second.size(); i++) {
                    ManagerSetupData setupData = GetPassInfoData(it->second[i]);
                    RegisterPassManagerSetup(i, setupData, pluginName);
                }
            } else {
                RegisterPluginEvent(inject, pluginName); // 注册event
            }
        }
        SetInjectFlag(true);
    } else {
        AddRegisteredUserFunc(value);
    }
}

void TimeoutFunc(union sigval sig)
{
    LOGW("client pid:%d timeout!\n", getpid());
    PluginClient::GetInstance()->SetUserFuncState(STATE_TIMEOUT);
}

void PluginClient::TimerStart(int interval)
{
    const int msTons = 1000000; // ms转ns倍数
    const int msTos = 1000; // s转ms倍数
    struct itimerspec time_value;
    time_value.it_value.tv_sec = (interval / msTos);
    time_value.it_value.tv_nsec = (interval % msTos) * msTons;
    time_value.it_interval.tv_sec = 0;
    time_value.it_interval.tv_nsec = 0;

    const int timeFlag = 0; // 0 表示希望timer首次到期时的时间与启动timer的时间间隔
    timer_settime(timerId, timeFlag, &time_value, NULL);
}

bool PluginClient::TimerInit(clockid_t id)
{
    struct sigevent evp;
    int sival = 124; // 传递整型参数，可以自定义
    memset(&evp, 0, sizeof(struct sigevent));
    evp.sigev_value.sival_ptr = timerId;
    evp.sigev_value.sival_int = sival;
    evp.sigev_notify = SIGEV_THREAD;
    evp.sigev_notify_function = TimeoutFunc;
    
    if (timer_create(id, &evp, &timerId) == -1) {
        LOGE("timer create fail\n");
        return false;
    }
    return true;
}

static bool WaitServer(const string& port)
{
    const int delay = 50;
    const int cnt = 4000;
    mode_t mask = umask(0);
    mode_t mode = 0666; // 权限是rwrwrw，跨进程时，其他用户也要可以访问
    string semFile = "wait_server_startup" + port;
    sem_t *sem = sem_open(semFile.c_str(), O_CREAT, mode, 0);
    umask(mask);
    int i = 0;
    for (; i < cnt; i++) {
        if (sem_trywait(sem) == 0) {
            break;
        } else {
            usleep(delay);
        }
    }
    sem_close(sem);
    sem_unlink(semFile.c_str());
    if (i >= cnt) {
        return false;
    }
    return true;
}

int PluginClient::ServerStart(pid_t& pid)
{
    if (!grpcPort.FindUnusedPort()) {
        LOGE("cannot find port for grpc,port 40001-65535 all used!\n");
        return -1;
    }

    int ret = 0;
    unsigned short portNum = grpcPort.GetPort();
    string port = std::to_string(portNum);
    pid = vfork();
    if (pid == 0) {
        LOGI("start plugin server!\n");
        string serverPath = input.GetServerPath();
        if (execl(serverPath.c_str(), port.c_str(), std::to_string(input.GetLogLevel()).c_str(), NULL) == -1) {
            DeleteGrpcPort();
            LOGE("server start fail! please check serverPath:%s\n", serverPath.c_str());
            ret = -1;
            _exit(0);
        }
    }

    if (!WaitServer(port)) {
        ret = -1;
    }
    return ret;
}

int PluginClient::ClientStart()
{
    setenv("no_grpc_proxy", "localhost", 1); // 关闭localhost的代理,因为server和client运行在同一个机器上，需要通过localhost建立连接
    string serverPort = "localhost:" + std::to_string(grpcPort.GetPort());
    grpcChannel = grpc::CreateChannel(serverPort, grpc::InsecureChannelCredentials());
    serviceStub = PluginService::NewStub(grpcChannel);
    SetInjectFlag(false);
    SetUserFuncState(STATE_WAIT_BEGIN);
    SetStartFlag(true);
    if (!TimerInit(CLOCK_REALTIME)) {
        return -1;
    }
    ReceiveSendMsg(grpckey[START], input.GetArgs());
    return 0;
}

void PluginClient::Init(struct plugin_name_args *pluginInfo, const string& pluginName, pid_t& serverPid)
{
    SetPluginName(pluginName);
    SetStartFlag(false);

    // inputCheck模块初始化,并对输入参数进行检查
    input.GetInputArgs(pluginInfo);
    if (input.GetInitInfo() != 0) {
        LOGD("read default info from pin-gcc-client.json fail! use the default timeout=%dms\n", input.GetTimeout());
    }
    if (input.GetServerPath() == "") {
        LOGE("server path is NULL!\n");
        return;
    }
    if (input.CheckSHA256() != 0) {
        LOGE("sha256 check sha256 file:%s fail!\n", input.GetShaPath().c_str());
        return;
    } else {
        LOGI("sha256 check success!\n");
    }

    // 查找未使用的端口号，并启动server和client
    if (ServerStart(serverPid) != 0) {
        DeleteGrpcPort();
        return;
    }
    if (ClientStart() != 0) {
        return;
    }

    // 等待用户完成注入点的注册或者server异常退出后，再接着执行gcc编译
    int status;
    while (1) {
        if ((GetInjectFlag()) || (GetUserFuncState() == STATE_TIMEOUT)) {
            break;
        }
        if (serverPid == waitpid(-1, &status, WNOHANG)) {
            DeleteGrpcPort();
            break;
        }
    }
}
} // namespace PinClient
