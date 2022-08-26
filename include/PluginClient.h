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
    This file contains the declaration of the PluginClient class.
*/

#ifndef GCC_PLUGIN_CLIENT_H
#define GCC_PLUGIN_CLIENT_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <time.h>
#include <signal.h>

#include <grpcpp/grpcpp.h>
#include "plugin.grpc.pb.h"
#include "gcc-plugin.h"
#include "PluginAPI/PluginAPI_Client.h"

using std::cout;
using std::string;
using std::endl;
using std::vector;

using plugin::PluginService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using plugin::ClientMsg;
using plugin::ServerMsg;
using Plugin_API::Operation;
using Plugin_API::Decl;
using Plugin_API::Type;

typedef std::function<void(plugin_event event, string& pluginName)> OptimizeFunc;
class PluginClient {
public:
    PluginClient(std::shared_ptr<Channel> channel) : stub_(PluginService::NewStub(channel)) {}
    void Optimize(const string& attribute, const string& value);
    static std::shared_ptr<PluginClient> GetInstance(void);
    void OperationJsonSerialize(vector<Operation>& data, string& out);
    void DeclJsonSerialize(Decl& decl, string& out);
    void TypeJsonSerialize(Type& data, string& out);
    void GetOptimizeResult(const string& funname, const string& param);
    static int GetInitInfo(string& path, int& timeout);
    static int CheckSHA256(const string& serverPath);
    static void CheckSafeCompile(const string& param);
    static void GetArg(struct plugin_name_args *pluginInfo, string& serverPath, string& arg);
    vector<ServerMsg> &GetMsg()
    {
        return serverMsg_;
    }
    void eraseMsg(int i);
    int GetUserFuncEnd(void)
    {
        return userFuncEnd;
    }
    void SetUserFuncEnd(int end)
    {
        userFuncEnd = end;
    }
    string &GetFuncName()
    {
        return funname;
    }
    void SetFuncName(const string& name)
    {
        funname = name;
    }
    string &GetFuncParam()
    {
        return param;
    }
    void SetFuncParam(const string& name)
    {
        param = name;
    }
    int GetOptimizeFlag(void)
    {
        return optimizeFlag;
    }
    void SetOptimizeFlag(int flag)
    {
        optimizeFlag = flag;
    }
    void SetTimeout(int time)
    {
        timeout_ = time;
    }
    void SetGccOptimizeFunc(OptimizeFunc func)
    {
        gccOptimizeFunc = func;
        injectFlag = 0;
    }
    OptimizeFunc GetGccOptimizeFunc(void)
    {
        return gccOptimizeFunc;
    }
    void SetPluginName(const string& pluginName)
    {
        this->pluginName = pluginName;
    }
    string GetPluginName(void)
    {
        return pluginName;
    }
    void SetInjectFlag(int flag)
    {
        injectFlag = flag;
    }
    int GetInjectFlag(void)
    {
        return injectFlag;
    }
    void TimerInit(void);
    void TimerStart(int interval);

private:
    std::unique_ptr<PluginService::Stub> stub_;
    vector<ServerMsg> serverMsg_;
    int userFuncEnd;
    string funname;
    string param;
    int optimizeFlag;
    int timeout_;
    timer_t timerId;
    OptimizeFunc gccOptimizeFunc;
    string pluginName;
    int injectFlag;
};

int ClientStart(int timeout, const string& arg, OptimizeFunc func, const string& pluginName);

#endif
