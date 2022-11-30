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

void PluginClient::TypeJsonSerialize (PluginIR::PluginTypeBase& type, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    uint64_t ReTypeId;
    uint64_t ReTypeWidth;

    ReTypeId = static_cast<uint64_t>(type.getPluginTypeID());
    item["id"] = std::to_string(ReTypeId);

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

    if (type.getReadOnlyFlag() == 1) {
        item["readonly"] = "1";
    }else {
        item["readonly"] = "0";
    }

    root["type"] = item;
    out = root.toStyledString();
}

void PluginClient::OpJsonSerialize(vector<FunctionOp>& data, string& out)
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
        item["attributes"]["funcName"] = d.funcNameAttr().getValue().data();
        operation = "operation" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
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
        OpJsonSerialize(allFuncOps, result);
        this->ReceiveSendMsg("FuncOpResult", result);
    } else if (funcName == "GetLocalDecls") {
        mlir::MLIRContext context;
        context.getOrLoadDialect<PluginDialect>();
        PluginAPI::PluginClientAPI clientAPI(context);
        uint64_t funcID = atol(root[std::to_string(0)].asString().c_str());
        vector<LocalDeclOp> decls = clientAPI.GetDecls(funcID);
        LocalDeclsJsonSerialize(decls, result);
        this->ReceiveSendMsg("LocalDeclOpResult", result);
    }else {
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
        LOGE("cannot find port for grpc\n");
        return -1;
    }

    port = std::to_string(portNum);
    pid = vfork();
    if (pid == 0) {
        LOGI("start plugin server!\n");
        string paramTimeout = std::to_string(timeout);
        if (execl(serverPath.c_str(), paramTimeout.c_str(), port.c_str(),
            std::to_string(logLevel).c_str(), NULL) == -1) {
            PluginClient::DeletePortFromLockFile(portNum);
            LOGE("server start fail! serverPath:%s\n", serverPath.c_str());
            _exit(0);
        }
    }
    int delay = 500000; // 500ms
    usleep(delay); // wait server start
    return 0;
}

int ClientStart(int timeout, const string& arg, const string& pluginName, const string& port)
{
    string serverPort = "localhost:" + port;
    g_grpcChannel = grpc::CreateChannel(serverPort, grpc::InsecureChannelCredentials());
    g_plugin = std::make_shared<PluginClient>(g_grpcChannel);
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
