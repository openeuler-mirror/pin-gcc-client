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

#include <thread>
#include <fstream>
#include <iostream>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <json/json.h>
#include "IRTrans/IRTransPlugin.h"

namespace PinClient {
using namespace Plugin_API;
using std::ios;
static std::shared_ptr<PluginClient> g_plugin = nullptr;

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

void PluginClient::IRTransBegin(const string& funcName, const string& param)
{
    PluginAPI_Client pluginAPI;
    string result;
    
    Json::Value root;
    Json::Reader reader;
    reader.parse(param, root);
    LOGD("%s func:%s,param:%s\n", __func__, funcName.c_str(), param.c_str());

    if (funcName == "SelectOperation") {
        int op = root["Opcode"].asInt();
        string attribute = root["string"].asString();
        vector<Operation> operations = pluginAPI.SelectOperation((Opcode)op, attribute);
        OperationJsonSerialize(operations, result);
        this->ReceiveSendMsg("OperationResult", result);
    } else if (funcName == "GetAllFunc") {
        string attribute = root["string"].asString();
        vector<Operation> operations = pluginAPI.GetAllFunc(attribute);
        OperationJsonSerialize(operations, result);
        this->ReceiveSendMsg("OperationResult", result);
    } else if (funcName == "SelectDeclByID") {
        string attribute = root["uintptr_t"].asString();
        uintptr_t id = atol(attribute.c_str());
        Decl decl = pluginAPI.SelectDeclByID(id);
        DeclJsonSerialize(decl, result);
        this->ReceiveSendMsg("DeclResult", result);
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
        LOGD("open %s fail! use default sha256file:%s\n", configFilePath.c_str(), shaPath.c_str());
        return -1;
    }
    reader.parse(ifs, root);
    ifs.close();

    if (serverPath == "") {
        serverPath = root["path"].asString();
        int index = serverPath.find_last_of("/");
        serverDir = serverPath.substr(0, index);
    }
    int timeoutJson = root["timeout"].asInt();
    if ((timeoutJson > 0) && (timeoutJson < 1000)) { // 不在0~1000ms范围内，使用默认值
        timeout = timeoutJson;
    } else {
        LOGW("timeout in config file should be 0~1000\n");
    }
    shaPath = root["sha256file"].asString();
    int ret = access(shaPath.c_str(), F_OK);
    if ((shaPath == "") || (ret != 0)) {
        shaPath = serverDir + "/libpin_user.sha256"; // sha256文件默认和server在同一目录
        LOGD("sha256 file not found,use default:%s\n", shaPath.c_str());
    }
    return 0;
}

void PluginClient::GetArg(struct plugin_name_args *pluginInfo, string& serverPath,
    string& arg, LogPriority& logLevel)
{
    Json::Value root;
    for (int i = 0; i < pluginInfo->argc; i++) {
        string key = pluginInfo->argv[i].key;
        if (key == "server_path") {
            serverPath = pluginInfo->argv[i].value;
        } else if (key == "log_level") {
            logLevel = (LogPriority)atoi(pluginInfo->argv[i].value);
            SetLogPriority(logLevel);
        } else {
            string value = pluginInfo->argv[i].value;
            CheckSafeCompileFlag(key, value);
            root[key] = value;
        }
    }
    arg = root.toStyledString();
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

    string cmd = "cd " + dir + " && " + "sha256sum -c " + filename;
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

void DeletePortInFile(unsigned short port)
{
    mode_t mask = umask(0);
    int fd = open("/tmp/file_lock_pin_client", O_CREAT | O_WRONLY, 0666);
    umask(mask);
    flock(fd, LOCK_EX);

    std::ifstream ifs;
    ifs.open("/tmp/gcc-client-port.txt");
    if (!ifs.is_open()) {
        LOGE("fs open fail\n");
        flock(fd, LOCK_UN);
        close(fd);
        return;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    ifs.close();

    string ports = buffer.str();
    string portStr = std::to_string(port) + "\n";
    int pos = ports.find(portStr);
    ports = ports.erase(pos, portStr.size());

    std::ofstream ofs;
    ofs.open("/tmp/gcc-client-port.txt", ios::trunc);
    if (!ofs.is_open()) {
        LOGE("fs open fail\n");
        flock(fd, LOCK_UN);
        close(fd);
        return;
    }
    ofs << ports;
    ofs.close();
    flock(fd, LOCK_UN);
    close(fd);
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

    ServerMsg serverMsg;
    while (stream->Read(&serverMsg)) {
        TimerStart(0);
        LOGD("rec from server:%s,%s\n", serverMsg.attribute().c_str(), serverMsg.value().c_str());
        if ((serverMsg.attribute() == "start") && (serverMsg.value() == "ok")) {
            DeletePortInFile(GetGrpcPort());
            LOGI("server has been started!\n");
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
                RegisterPluginEvent(inject, pluginName); // 注册event
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
    LOGW("client timeout!\n");
    PluginClient::GetInstance()->SetUserFuncState(STATE_TIMEOUT);
}

void PluginClient::TimerStart(int interval)
{
    int msTons = 1000000;
    struct itimerspec time_value;
    time_value.it_value.tv_sec = 0;
    time_value.it_value.tv_nsec = interval * msTons;
    time_value.it_interval.tv_sec = 0;
    time_value.it_interval.tv_nsec = 0;
    
    timer_settime(timerId, 0, &time_value, NULL);
}

void PluginClient::TimerInit(void)
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
    }
}

unsigned short PluginClient::FindUnusedPort(void)
{
    unsigned short basePort = 40000; // grpc通信端口号从40000开始
    unsigned short foundPort = 0;
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr("0.0.0.0");
    std::ifstream ifs;
    std::ofstream ofs;
    
    mode_t mask = umask(0);
    int fd = open("/tmp/file_lock_pin_client", O_CREAT | O_WRONLY, 0666);
    ofs.open("/tmp/gcc-client-port.txt", ios::app);
    umask(mask);
    flock(fd, LOCK_EX);
    ifs.open("/tmp/gcc-client-port.txt");
    if (!ofs.is_open()) {
        LOGE("ofs open /tmp/gcc-client-port.txt fail\n");
        return 0;
    }
    if (!ifs.is_open()) {
        LOGE("ifs open /tmp/gcc-client-port.txt fail\n");
        return 0;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    string buf = buffer.str();
    ifs.close();

    while (++basePort < UINT16_MAX) {
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        serverAddr.sin_port = htons(basePort);
        int ret = connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
        if (sock != -1) {
            close(sock);
        }
        if ((ret == -1) && (errno == ECONNREFUSED)) {
            string strPort = std::to_string(basePort) + "\n";
            if (buf.find(strPort) == buf.npos) {
                foundPort = basePort;
                ofs << strPort;
                break;
            }
        }
    }
    ofs.close();
    if (basePort == UINT16_MAX) {
        ofs.open("/tmp/gcc-client-port.txt", ios::trunc);
        ofs.close();
    }
    
    flock(fd, LOCK_UN);
    close(fd);
    return foundPort;
}

int ServerStart(int timeout, const string& serverPath, pid_t& pid, string& port, const LogPriority logLevel)
{
    unsigned short portNum = PluginClient::FindUnusedPort();
    if (portNum == 0) {
        LOGE("cannot find port for grpc\n");
        return -1;
    }

    port = std::to_string(portNum);
    pid = fork();
    if (pid == 0) {
        LOGI("start plugin server!\n");
        string paramTimeout = std::to_string(timeout);
        if (execl(serverPath.c_str(), paramTimeout.c_str(), port.c_str(),
            std::to_string(logLevel).c_str(), NULL) == -1) {
            LOGE("server start fail! serverPath:%s\n", serverPath.c_str());
            exit(0);
        }
    }
    int delay = 100000; // 100ms
    usleep(delay); // wait server start
    return 0;
}

int ClientStart(int timeout, const string& arg, const string& pluginName, const string& port)
{
    string serverPort = "localhost:" + port;
    g_plugin = std::make_shared<PluginClient>(
        grpc::CreateChannel(serverPort, grpc::InsecureChannelCredentials()));
    g_plugin->SetInjectFlag(false);
    g_plugin->SetTimeout(timeout);
    g_plugin->SetUserFuncState(STATE_WAIT_BEGIN);
    g_plugin->TimerInit();
    unsigned short grpcPort = (unsigned short)atoi(port.c_str());
    g_plugin->SetGrpcPort(grpcPort);
    g_plugin->ReceiveSendMsg("start", arg);
    return 0;
}
} // namespace PinClient
