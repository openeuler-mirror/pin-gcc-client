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
    主要完成功能：完成和server之间grpc通信及数据解析，获取gcc插件数据并进行IR转换，完成
    gcc注册点注入及参数保存。提供GetInstance获取client对象唯一实例，完成插件初始化并启动
    server子进程，处理超时异常事件
*/

#ifndef PLUGIN_CLIENT_H
#define PLUGIN_CLIENT_H

#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"
#include <grpcpp/grpcpp.h>
#include "plugin.grpc.pb.h"
#include "PluginAPI/PluginClientAPI.h"
#include "PluginClient/PluginGrpcPort.h"
#include "PluginClient/PluginInputCheck.h"
#include "PluginClient/PluginJson.h"
#include "PluginClient/PluginLog.h"
#include "Dialect/PluginDialect.h"

namespace PinClient {
using plugin::PluginService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using plugin::ClientMsg;
using plugin::ServerMsg;

enum Grpckey {
#define GRPC_KEY(KEY, NAME) KEY,
#include "GrpcKey.def"
#undef GRPC_KEY
MAX_GRPC_KEYS
};

#define GRPC_KEY(KEY, NAME) NAME,
const char *const grpckey[] = {
#include "GrpcKey.def"
};
#undef GRPC_KEY

enum GrpcValue {
#define GRPC_VALUE(KEY, NAME) KEY,
#include "GrpcValue.def"
#undef GRPC_VALUE
MAX_GRPC_VALUES
};

#define GRPC_VALUE(KEY, NAME) NAME,
const char *const grpcValue[] = {
#include "GrpcValue.def"
};
#undef GRPC_VALUE

enum PassValue {
#define PASS_VALUE(KEY, NAME) KEY,
#include "PassValue.def"
#undef PASS_VALUE
MAX_PASS_VALUES
};

#define PASS_VALUE(KEY, NAME) NAME,
const char *const passValue[] = {
#include "PassValue.def"
};
#undef PASS_VALUE

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
    HANDLE_INCLUDE_FILE,
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
    PASS_MAC,
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
    /* 定义的grpc服务端和客户端通信的接口函数 */
    void ReceiveSendMsg(const string& attribute, const string& value);
    /* 获取client对象实例,有且只有一个实例对象 */
    static PluginClient *GetInstance();
    /* 获取gcc插件数据并进行IR转换，将转换后的数据序列化返回给server。param：函数入参序列化后的数据 */
    void GetIRTransResult(void *gccData, const string& funname, const string& param);
    void GetGccData(const string& funcName, const string& param, string& key, string& result);
    /* 将服务端传递的InjectPoint转换为plugin_event */
    static int GetEvent(InjectPoint inject, plugin_event *event);
    void Init(struct plugin_name_args *pluginInfo, const string& pluginName, pid_t& serverPid);
    UserFuncStateEnum GetUserFuncState()
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
    bool TimerInit(clockid_t id);
    void TimerStart(int interval);
    /* 保存注入点和函数名信息,value格式为 注入点:函数名称 */
    int AddRegisteredUserFunc(const string& value);
    map<InjectPoint, vector<string>>& GetRegisteredUserFunc()
    {
        return registeredUserFunc;
    }
    vector<string>& GetFuncNameByInject(InjectPoint inject)
    {
        return registeredUserFunc[inject];
    }
    /* grpc消息处理函数 */
    void ServerMsgProc(const string& attribute, const string& value);
    int ClientStart();
    int ServerStart(pid_t& pid); // pid server线程pid
    bool DeleteGrpcPort()
    {
        return grpcPort.DeletePortFromLockFile();
    }
    bool GetStartFlag()
    {
        return startFlag;
    }
    void SetStartFlag(bool flag)
    {
        startFlag = flag;
    }
    PluginJson &GetJson(void)
    {
        return json;
    }

private:
    std::unique_ptr<PluginService::Stub> serviceStub; // 保存grpc客户端stub对象
    /* server将函数名和参数发给client，client将数据返回server后，server返回completed消息，然后接着向client发送下一个函数,
       client对应函数的开始执行由server消息控制，需要结束标志告诉等待线程结束等待，执行下一函数 */
    volatile UserFuncStateEnum userFuncState;
    string pluginAPIName; // 保存用户调用PluginAPI的函数名
    string pluginAPIParams; // 保存用户调用PluginAPI函数的参数
    timer_t timerId;
    string pluginName; // 向gcc插件注册回调函数时需要
    bool injectFlag; // 是否完成将注册点信息注册到gcc
    PluginGrpcPort grpcPort;
    PluginInputCheck input;
    PluginJson json;
    /* 保存注册点和函数信息 */
    map<InjectPoint, vector<string>> registeredUserFunc;
    std::shared_ptr<Channel> grpcChannel;
    bool startFlag;
};
} // namespace PinClient

#endif
