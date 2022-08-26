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
    This file contains the declaration of the PluginAPI class.
*/

#include <map>
#include <set>
#include <vector>
#include <string>
#include <json/json.h>
#include "PluginClient.h"
#include "gcc-plugin.h"
#include "plugin-version.h"
#include "PluginAPI/PluginAPI_Client.h"
#include "tree-pass.h"
#include "tree.h"
#include "Plugin_Log.h"

using namespace std;
using namespace Plugin_API;
using namespace Plugin_Client_Log;

using std::vector;
int plugin_is_GPL_compatible;

static void GccEnd(void *gccData, void *userData)
{
    printf("gcc optimize has been done! now close server...\n");
    PluginClient::GetInstance()->Optimize("stop", "");
}

static void GccOptimizeFunc(void *gccData, void *userData)
{
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    while (1) {
        if (client->GetUserFuncEnd() == 1) {
            break;
        }

        string funname = client->GetFuncName();
        string param = client->GetFuncParam();
        if (funname != "") {
            client->SetOptimizeFlag(1);
            client->GetOptimizeResult(funname, param);
            client->SetOptimizeFlag(0);
        }
    }
}

void GccOptimize(plugin_event event, string& pluginName)
{
    register_callback(pluginName.c_str(), event, &GccOptimizeFunc, NULL);
    register_callback(pluginName.c_str(), PLUGIN_FINISH, &GccEnd, NULL);
}

int plugin_init(struct plugin_name_args *pluginInfo, struct plugin_gcc_version *version)
{
    if (!plugin_default_version_check(version, &gcc_version)) {
        printf("incompatible gcc/plugin versions\n");
        return 1;
    }
    string pluginName = pluginInfo->base_name;

    int timeout = 100;
    string path = "";
    if (PluginClient::GetInitInfo(path, timeout) != 0) {
        printf("read timeout from config.json fail! use the default timeout=100ms\n");
    }

    string serverPath = "";
    string arg = "";
    PluginClient::GetArg(pluginInfo, serverPath, arg);
    if (serverPath == "") {
        if (path != "") {
            serverPath = path;
        } else {
            printf("server path is NULL!\n");
            return 0;
        }
    }

    if (PluginClient::CheckSHA256(serverPath) != 0) {
        printf("sha256 check fail!\n");
        return 0;
    } else {
        printf("sha256 check success!\n");
    }

    pid_t pid = fork();
    if (pid == 0) {
        printf("start plugin server!\n");
        string para = std::to_string(timeout);
        execl(serverPath.c_str(), para.c_str(), NULL);
    }
    int delay = 500000;
    usleep(delay); // wait server start

    int status;
    ClientStart(timeout, arg, GccOptimize, pluginName);
    while (1) {
        if (PluginClient::GetInstance()->GetInjectFlag() == 1)
            break;
        if (pid == waitpid(-1, &status, WNOHANG)) {
            break;
        }
    }
    return 0;
}

