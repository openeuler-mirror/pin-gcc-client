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
    This file contains the declaration of the PluginClient class.
*/

#ifndef PLUGIN_CLIENT_H
#define PLUGIN_CLIENT_H

#include <memory>
#include <string>
#include <vector>
#include <time.h>
#include <signal.h>
#include <json/json.h>

#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"
#include <grpcpp/grpcpp.h>
#include "plugin.grpc.pb.h"
#include "gcc-plugin.h"
#include "PluginAPI/PluginClientAPI.h"
#include "PluginClient/PluginLog.h"

namespace PinClient {
using std::cout;
using std::string;
using std::endl;
using std::vector;
using std::map;
using std::pair;

using plugin::PluginService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using plugin::ClientMsg;
using plugin::ServerMsg;

enum InjectPoint : uint8_t {
    HANDLE_PARSE_TYPE = 0,
    HANDLE_PARSE_DECL,
    HANDLE_PRAGMAS,
    HANDLE_PARSE_FUNCTION,
    HANDLE_BEFORE_IPA,
    HANDLE_AFTER_IPA,
    HANDLE_BEFORE_EVERY_PASS,
    HANDLE_AFTER_EVERY_PASS,
    HANDLE_BEFORE_ALL_PASS,
    HANDLE_AFTER_ALL_PASS,
    HANDLE_COMPILE_END,
    HANDLE_MANAGER_SETUP,
    HANDLE_MAX,
};
typedef enum {
    STATE_WAIT_BEGIN = 0,
    STATE_BEGIN,
    STATE_WAIT_IR, // 等待IR转换完成
    STATE_RETURN, // 已完成IR转换并将数据返回给server
    STATE_END, // 用户当前函数已执行完毕
    STATE_TIMEOUT, // server进程异常,接收不到server数据，超时
} UserFuncStateEnum;

// 参考点名称
enum RefPassName {
    PASS_CFG,
    PASS_PHIOPT,
    PASS_SSA,
    PASS_LOOP,
};

enum PassPosition {
    PASS_INSERT_AFTER,
    PASS_INSERT_BEFORE,
    PASS_REPLACE,
};

struct ManagerSetupData {
    RefPassName refPassName;
    int passNum; // 指定passName的第几次执行作为参考点
    PassPosition passPosition; // 指定pass是添加在参考点之前还是之后
};

class PluginClient {
public:
    PluginClient() = default;
    ~PluginClient() = default;
    PluginClient(std::shared_ptr<Channel> channel) : serviceStub(PluginService::NewStub(channel)) {}
    /* 定义的grpc服务端和客户端通信的接口函数 */
    void ReceiveSendMsg(const string& attribute, const string& value);
    /* 获取client对象实例,有且只有一个实例对象 */
    static std::shared_ptr<PluginClient> GetInstance(void);
    void OpJsonSerialize(vector<mlir::Plugin::FunctionOp>& data, string& out);
    void LoopOpsJsonSerialize(vector<mlir::Plugin::LoopOp>& loops, string& out);
    void LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out);
    void BlocksJsonSerialize(vector<uint64_t>&, string&);
    void EdgesJsonSerialize(vector<pair<uint64_t, uint64_t> >&, string&);
    void EdgeJsonSerialize(pair<uint64_t, uint64_t>&, string&);
    void NopJsonSerialize(string&);
    void FunctionOpJsonSerialize(vector<mlir::Plugin::FunctionOp>& data, string& out);
    void LocalDeclsJsonSerialize(vector<mlir::Plugin::LocalDeclOp>& decls, string& out);
    Json::Value OperationJsonSerialize(mlir::Operation *, uint64_t&);
    Json::Value CallOpJsonSerialize(mlir::Plugin::CallOp& data);
    Json::Value CondOpJsonSerialize(mlir::Plugin::CondOp& data, uint64_t&);
    Json::Value PhiOpJsonSerialize(mlir::Plugin::PhiOp& data);
    Json::Value AssignOpJsonSerialize(mlir::Plugin::AssignOp& data);
    Json::Value BaseOpJsonSerialize(mlir::Plugin::BaseOp data);
    Json::Value FallThroughOpJsonSerialize(mlir::Plugin::FallThroughOp data, uint64_t&);
    Json::Value RetOpJsonSerialize(mlir::Plugin::RetOp data, uint64_t&);
    Json::Value ValueJsonSerialize(mlir::Value value);
    /* 将Type类型数据序列化 */
    Json::Value TypeJsonSerialize(PluginIR::PluginTypeBase& type);
    /* 获取gcc插件数据并进行IR转换，将转换后的数据序列化返回给server。param：函数入参序列化后的数据 */
    void IRTransBegin(const string& funname, const string& param);
    /* 从配置文件读取初始化信息 */
    static int GetInitInfo(string& serverPath, string& shaPath, int& timeout);
    /* 进行sha256校验 */
    static int CheckSHA256(const string& shaPath);
    static void CheckSafeCompileFlag(const string& argName, const string& param);
    /* 解析gcc编译时传递的-fplugin-arg参数 */
    static void GetArg(struct plugin_name_args *pluginInfo, string& serverPath, string& arg, LogPriority& logLevel);
    /* 将服务端传递的InjectPoint转换为plugin_event */
    static int GetEvent(InjectPoint inject, plugin_event *event);
    static unsigned short FindUnusedPort(void); // 查找未被使用的端口号，确保并发情况下server和client一对一
    UserFuncStateEnum GetUserFuncState(void)
    {
        return userFuncState;
    }
    void SetUserFuncState(UserFuncStateEnum state)
    {
        userFuncState = state;
    }
    string &GetPluginAPIName(void)
    {
        return pluginAPIName;
    }
    void SetPluginAPIName(const string& name)
    {
        pluginAPIName = name;
    }
    string &GetPluginAPIParam(void)
    {
        return pluginAPIParams;
    }
    void SetPluginAPIParam(const string& name)
    {
        pluginAPIParams = name;
    }
    void SetTimeout(int time)
    {
        timeout = time;
    }
    void SetPluginName(const string& pluginName)
    {
        this->pluginName = pluginName;
    }
    string GetPluginName(void)
    {
        return pluginName;
    }
    void SetInjectFlag(bool flag)
    {
        injectFlag = flag;
    }
    bool GetInjectFlag(void)
    {
        return injectFlag;
    }
    void SetGrpcPort(unsigned short port)
    {
        grpcPort = port;
    }
    unsigned short GetGrpcPort(void)
    {
        return grpcPort;
    }
    void TimerInit(void);
    void TimerStart(int interval);
    /* 保存注入点和函数名信息,value格式为 注入点:函数名称 */
    int AddRegisteredUserFunc(const string& value);
    map<InjectPoint, vector<string>>& GetRegisteredUserFunc(void)
    {
        return registeredUserFunc;
    }
    vector<string>& GetFuncNameByInject(InjectPoint inject)
    {
        return registeredUserFunc[inject];
    }
    /* grpc消息处理函数 */
    void ServerMsgProc(const string& attribute, const string& value);
    /* grpc的server被client拉起之前将port记录在/tmp/grpc_ports_pin_client.txt中, server和client建立通信后从文件中删除port，避免多进程时端口冲突
       文件若不存在，先创建文件 */
    static int OpenLockFile(const char *path);
    /* 读取文件中保存的grpc端口号 */
    static void ReadPortsFromLockFile(int fd, string& grpcPorts);
    /* server启动异常或者grpc建立通信后,将文件中记录的端口号删除 */
    static bool DeletePortFromLockFile(unsigned short port);

private:
    std::unique_ptr<PluginService::Stub> serviceStub; // 保存grpc客户端stub对象
    /* server将函数名和参数发给client，client将数据返回server后，server返回completed消息，然后接着向client发送下一个函数,
       client对应函数的开始执行由server消息控制，需要结束标志告诉等待线程结束等待，执行下一函数 */
    volatile UserFuncStateEnum userFuncState;
    string pluginAPIName; // 保存用户调用PluginAPI的函数名
    string pluginAPIParams; // 保存用户调用PluginAPI函数的参数
    int timeout;
    timer_t timerId;
    string pluginName; // 向gcc插件注册回调函数时需要
    bool injectFlag; // 是否完成将注册点信息注册到gcc
    unsigned short grpcPort; // server和client使用的端口号
    /* 保存注册点和函数信息 */
    map<InjectPoint, vector<string>> registeredUserFunc;
};

/* pid:子进程server返回的pid，port:查找到的未使用的端口号 */
int ServerStart(int timeout, const string& serverPath, pid_t& pid, string& port, const LogPriority logLevel);
int ClientStart(int timeout, const string& arg, const string& pluginName, const string& port);
} // namespace PinClient

#endif
