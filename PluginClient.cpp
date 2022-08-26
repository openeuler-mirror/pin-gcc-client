/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may
   not use this file except in compliance with the License. You may obtain
   a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
   License for the specific language governing permissions and limitations
   under the License.

   Author: Mingchuan Wu and Yancheng Li
   Create: 2022-08-18
   Description:
    This file contains the implementation of the PluginClient class..
*/


#include <thread>
#include <fstream>
#include <semaphore.h>
#include <json/json.h>
#include "Plugin_Log.h"
#include "PluginClient.h"

using namespace Plugin_API;
using namespace Plugin_Client_Log;
static sem_t g_sem;
static std::shared_ptr<PluginClient> g_plugin = nullptr;

std::map<std::string, plugin_event> g_inject_point {
    {"HANDLE_PARSE_TYPE", PLUGIN_FINISH_TYPE},
    {"HANDLE_PARSE_DECL", PLUGIN_FINISH_DECL},
    {"HANDLE_PRAGMA", PLUGIN_PRAGMAS},
    {"HANDLE_PARSE_FUNCTION", PLUGIN_FINISH_PARSE_FUNCTION},
    {"HANDLE_BEFORE_IPA", PLUGIN_ALL_IPA_PASSES_START},
    {"HANDLE_AFTER_IPA", PLUGIN_ALL_IPA_PASSES_END},
    {"HANDLE_BEFORE_EVERY_PASS", PLUGIN_EARLY_GIMPLE_PASSES_START},
    {"HANDLE_AFTER_EVERY_PASS", PLUGIN_EARLY_GIMPLE_PASSES_END},
    {"HANDLE_BEFORE_ALL_PASS", PLUGIN_ALL_PASSES_START},
    {"HANDLE_AFTER_ALL_PASS", PLUGIN_ALL_PASSES_END},
    {"HANDLE_COMPILE_END", PLUGIN_FINISH},
};

std::shared_ptr<PluginClient> PluginClient::GetInstance(void)
{
    return g_plugin;
}

static int GetEvent(const string& s, plugin_event *event)
{
    auto it = g_inject_point.find(s);
    if (it != g_inject_point.end()) {
        *event = it->second;
        return 0;
    }
    return -1;
}

void PluginClient::OperationJsonSerialize(vector<Operation>& data, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    int i = 0;
    string operation;
    
    for (auto& d: data) {
        item["id"] = std::to_string(d.GetID());
        item["opCode"] = d.GetOpcode();
        for (map<string, string>::reverse_iterator iter = d.GetAttributes().rbegin();
            iter != d.GetAttributes().rend(); iter++) {
            item["attributes"][iter->first] = iter->second;
        }

        item["resultType"]["id"] = std::to_string(d.GetResultTypes().GetID());
        item["resultType"]["typeCode"] = d.GetResultTypes().GetTypeCode();
        item["resultType"]["tQual"] = d.GetResultTypes().GetTQual();
        for (map<string, string>::reverse_iterator result = d.GetResultTypes().GetAttributes().rbegin();
            result != d.GetResultTypes().GetAttributes().rend(); result++) {
            item["resultType"]["attributes"][result->first] = result->second;
        }

        Decl decl;
        for (map<string, Decl>::reverse_iterator operand = d.GetOperands().rbegin();
            operand != d.GetOperands().rend(); operand++) {
            decl = operand->second;
            item["operands"][operand->first]["id"] = std::to_string(decl.GetID());
            item["operands"][operand->first]["declCode"] = decl.GetDeclCode();
            for (map<string, string>::reverse_iterator att = decl.GetAttributes().rbegin();
                att != decl.GetAttributes().rend(); att++) {
                item["operands"][operand->first]["attributes"][att->first] = att->second;
            }

            item["operands"][operand->first]["declType"]["id"] = std::to_string(decl.GetType().GetID());
            item["operands"][operand->first]["declType"]["typeCode"] = decl.GetType().GetTypeCode();
            item["operands"][operand->first]["declType"]["tQual"] = decl.GetType().GetTQual();
            for (map<string, string>::reverse_iterator decls = decl.GetType().GetAttributes().rbegin();
                decls != decl.GetType().GetAttributes().rend(); decls++) {
                item["operands"][operand->first]["declType"]["attributes"][decls->first] = decls->second;
            }
        }

        operation = "operation" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginClient::DeclJsonSerialize(Decl& decl, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    item["id"] = std::to_string(decl.GetID());
    item["declCode"] = decl.GetDeclCode();
    for (map<string, string>::reverse_iterator declAtt = decl.GetAttributes().rbegin();
        declAtt != decl.GetAttributes().rend(); declAtt++) {
        item["attributes"][declAtt->first] = declAtt->second;
    }

    item["declType"]["id"] = std::to_string(decl.GetType().GetID());
    item["declType"]["typeCode"] = decl.GetType().GetTypeCode();
    item["declType"]["tQual"] = decl.GetType().GetTQual();
    map<string, string> tmp = decl.GetType().GetAttributes();
    for (map<string, string>::reverse_iterator typeAtt = tmp.rbegin(); typeAtt != tmp.rend(); typeAtt++) {
        item["declType"]["attributes"][typeAtt->first] = typeAtt->second;
    }
    root["decl"] = item;
    out = root.toStyledString();
}

void PluginClient::TypeJsonSerialize(Type& type, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    item["id"] = std::to_string(type.GetID());
    item["typeCode"] = type.GetTypeCode();
    item["tQual"] = type.GetTQual();
    map<string, string> tmp = type.GetAttributes();
    for (map<string, string>::reverse_iterator typeAtt = tmp.rbegin(); typeAtt != tmp.rend(); typeAtt++) {
        item["attributes"][typeAtt->first] = typeAtt->second;
    }
    root["type"] = item;
    out = root.toStyledString();
}

void PluginClient::GetOptimizeResult(const string& funname, const string& param)
{
    PluginAPI_Client pluginAPI;
    string result;
    
    Json::Value root;
    Json::Reader reader;
    reader.parse(param, root);
    LOG("%s func:%s,param:%s\n", __func__, funname.c_str(), param.c_str());

    if (funname == "SelectOperation") {
        int op = root["Opcode"].asInt();
        string attribute = root["string"].asString();
        vector<Operation> operations = pluginAPI.SelectOperation((Opcode)op, attribute);
        OperationJsonSerialize(operations, result);
        this->Optimize("OperationResult", result);
    } else if (funname == "GetAllFunc") {
        string attribute = root["string"].asString();
        vector<Operation> operations = pluginAPI.GetAllFunc(attribute);
        OperationJsonSerialize(operations, result);
        this->Optimize("OperationResult", result);
    } else if (funname == "SelectDeclByID") {
        string attribute = root["uintptr_t"].asString();
        uintptr_t id = atol(attribute.c_str());
        Decl decl = pluginAPI.SelectDeclByID(id);
        DeclJsonSerialize(decl, result);
        this->Optimize("DeclResult", result);
    } else {
        printf("function: %s not found!\n", funname.c_str());
    }

    LOG("IR function: %s\n", funname.c_str());
    this->SetFuncName("");
    this->Optimize(funname, "done");
}

int PluginClient::GetInitInfo(string& path, int& timeout)
{
    Json::Value root;
    Json::Reader reader;
    std::ifstream ifs;
    ifs.open("config.json");
    if (!ifs) {
        printf("open config.json fail!\n");
        return -1;
    }
    reader.parse(ifs, root);
    ifs.close();

    path = root["path"].asString();
    timeout = root["timeout"].asInt();
    return 0;
}

void PluginClient::GetArg(struct plugin_name_args *pluginInfo, string& serverPath, string& arg)
{
    Json::Value root;
    for (int i = 0; i < pluginInfo->argc; i++) {
        string key = pluginInfo->argv[i].key;
        if (key == "server_path") {
            serverPath = pluginInfo->argv[i].value;
        } else {
            string value = pluginInfo->argv[i].value;
            CheckSafeCompile(value);
            root[key] = value;
        }
    }
    arg = root.toStyledString();
}

int PluginClient::CheckSHA256(const string& serverPath)
{
    int index = serverPath.find_last_of("/");
    string dir = serverPath.substr(0, index);
    string filename = serverPath.substr(index+1, -1);

    string cmd = "cd " + dir + " && " + "sha256sum -c " + filename + ".sha256";
    int ret = system(cmd.c_str());
    return ret;
}

void PluginClient::CheckSafeCompile(const string& param)
{
    vector<string> safeCompile = {
        "-z noexecstack",
        "-fno-stack-protector",
        "-fstack-protector-all",
        "-D_FORTIFY_SOURCE=1",
    };

    for (auto& v : safeCompile) {
        if (param.find(v) != string::npos) {
            printf("Warning: compile parameter:%s not safe!!!\n", param.c_str());
        }
    }
}

void PluginClient::Optimize(const string& attribute, const string& value)
{
    ClientContext context;
    auto stream = stub_->Optimize(&context);
    
    ClientMsg clientMsg;
    clientMsg.set_attribute(attribute);
    clientMsg.set_value(value);
    stream->Write(clientMsg);
    stream->WritesDone();
    TimerStart(timeout_);

    ServerMsg serverMsg;
    while (stream->Read(&serverMsg)) {
        TimerStart(0);
        LOG("rec from server:%s,%s\n", serverMsg.attribute().c_str(), serverMsg.value().c_str());
        if ((serverMsg.attribute() == "start") && (serverMsg.value() == "ok")) {
            cout << "server has been started!" << endl;
        } else if ((serverMsg.attribute() == "stop") && (serverMsg.value() == "ok")) {
            cout << "server has been closed!" << endl;
            Status status = stream->Finish();
            if (!status.ok()) {
                cout << status.error_code() << ": " << status.error_message() << endl;
                cout << "RPC failed";
            }
            CloseLog();
        } else if ((serverMsg.attribute() == "userFuncEnd") && (serverMsg.value() == "end")) {
            userFuncEnd = 1;
        } else {
            serverMsg_.push_back(serverMsg);
            sem_post(&g_sem);
        }
    }
}

void PluginClient::eraseMsg(int i)
{
    vector<ServerMsg>::iterator it = serverMsg_.begin() + i;
    serverMsg_.erase(it);
}

static void ServerMsgProc(std::shared_ptr<PluginClient> client, const string& attribute, const string& value)
{
    if (attribute == "injectPoint") {
        plugin_event event;
        if (GetEvent(value, &event) == 0) {
            OptimizeFunc func = client->GetGccOptimizeFunc();
            string pluginName = client->GetPluginName();
            func(event, pluginName);
            client->Optimize(attribute, "done");
        } else {
            client->Optimize(attribute, "error");
            client->Optimize("stop", "");
        }
        client->SetInjectFlag(1);
    } else {
        while (client->GetOptimizeFlag());
        client->SetFuncName(attribute);
        client->SetFuncParam(value);
    }
}

static void ParseServerMsg(void)
{
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    while (1) {
        sem_wait(&g_sem);
        vector<ServerMsg> serverMsg = client->GetMsg();
        for (unsigned int i = 0; i < serverMsg.size(); i++) {
            string attribute = serverMsg[i].attribute();
            string value = serverMsg[i].value();
            ServerMsgProc(client, attribute, value);
            client->eraseMsg(i);
        }
    }
}

static void TimeoutFunc(union sigval sig)
{
    printf("client timeout!\n");
    PluginClient::GetInstance()->SetUserFuncEnd(1);
}

void PluginClient::TimerStart(int interval)
{
    int msTons = 1000000;
    struct itimerspec time_value;
    time_value.it_value.tv_sec = 0;
    time_value.it_value.tv_nsec = interval * msTons;
    time_value.it_interval.tv_sec = 0;
    time_value.it_interval.tv_nsec = 0;
    
    timer_settime(&timerId, 0, &time_value, NULL);
}

void PluginClient::TimerInit(void)
{
    struct sigevent evp;
    int sival = 124;
    memset(&evp, 0, sizeof(struct sigevent));
    evp.sigev_value.sival_ptr = timerId;
    evp.sigev_value.sival_int = sival;
    evp.sigev_notify = SIGEV_THREAD;
    evp.sigev_notify_function = TimeoutFunc;
    
    if (timer_create(CLOCK_REALTIME, &evp, &timerId) == -1) {
        printf("timer create fail\n");
    }
}

int ClientStart(int timeout, const string& arg, OptimizeFunc func, const string& pluginName)
{
    sem_init(&g_sem, 0, 0);
    g_plugin = std::make_shared<PluginClient>(
        grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
    g_plugin->SetOptimizeFlag(0);
    g_plugin->SetGccOptimizeFunc(func);
    g_plugin->SetTimeout(timeout);
    g_plugin->TimerInit();
    g_plugin->Optimize("start", arg);

    std::thread parseServerMsgThread(ParseServerMsg);
    parseServerMsgThread.detach();
    
    return 0;
}
