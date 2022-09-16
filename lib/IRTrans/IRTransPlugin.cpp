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
    This file contains the declaration of the PluginAPI class.
*/

#include <map>
#include <set>
#include <vector>
#include <string>
#include <json/json.h>
#include "PluginClient/PluginClient.h"
#include "plugin-version.h"
#include "PluginAPI/PluginAPI_Client.h"
#include "tree-pass.h"
#include "tree.h"
#include "IRTrans/IRTransPlugin.h"

using namespace std;
using namespace Plugin_API;
using namespace PinClient;

using std::vector;
int plugin_is_GPL_compatible;

/* gcc插件end事件回调函数 */
static void GccEnd(void *gccData, void *userData)
{
    LOGI("gcc optimize has been done! now close server...\n");
    PluginClient::GetInstance()->ReceiveSendMsg("stop", "");
}

/* gcc插件回调函数,当注册的plugin_event触发时,进入此函数 */
static void GccEventCallback(void *gccData, void *userData)
{
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    InjectPoint *inject = (InjectPoint *)userData;
    vector<string> userFuncs = client->GetFuncNameByInject(*inject);
    string key = "injectPoint";
    string value;
    for (auto &userFunc : userFuncs) {
        if (client->GetUserFuncState() == STATE_TIMEOUT) {
            break;
        }
        value = std::to_string(*inject) + ":" + userFunc;
        client->ReceiveSendMsg(key, value);
        while (1) {
            UserFuncStateEnum state = client->GetUserFuncState();
            /* server获取到client对应函数的执行结果后,向client回复已执行完,跳出循环执行下一个函数 */
            if (state == STATE_END) {
                client->SetUserFuncState(STATE_WAIT_BEGIN);
                break;
            } else if (state == STATE_TIMEOUT) {
                break;
            } else if (state == STATE_BEGIN) {
                string funcName = client->GetPluginAPIName();
                string param = client->GetPluginAPIParam();
                if (funcName != "") {
                    client->SetUserFuncState(STATE_WAIT_IR);
                    client->IRTransBegin(funcName, param);
                }
            }
        }
    }
}

/* g_event作为userData传递给EventCallback */
static InjectPoint g_event[] = {
    HANDLE_PARSE_TYPE,
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
    HANDLE_MAX
};

int RegisterPluginEvent(InjectPoint inject, const string& pluginName)
{
    plugin_event event;
    if (PluginClient::GetEvent(inject, &event) != 0) {
        return -1;
    }
    register_callback(pluginName.c_str(), event, &GccEventCallback, (void *)&g_event[inject]);
    return 0;
}

int plugin_init(struct plugin_name_args *pluginInfo, struct plugin_gcc_version *version)
{
    if (!plugin_default_version_check(version, &gcc_version)) {
        LOGE("incompatible gcc/plugin versions\n");
        return 1;
    }
    string pluginName = pluginInfo->base_name;

    int timeout = 200; // 默认超时时间200ms
    string shaPath;
    string serverPath = "";
    string arg = "";
    LogPriority logLevel = PRIORITY_WARN;
    PluginClient::GetArg(pluginInfo, serverPath, arg, logLevel);
    if (PluginClient::GetInitInfo(serverPath, shaPath, timeout) != 0) {
        LOGD("read default info from pin-gcc-client.json fail! use the default timeout=%dms\n", timeout);
    }
    if (serverPath == "") {
        LOGE("server path is NULL!\n");
        return 0;
    }

    if (PluginClient::CheckSHA256(shaPath) != 0) {
        LOGE("sha256 check sha256 file:%s fail!\n", shaPath.c_str());
        return 0;
    } else {
        LOGD("sha256 check success!\n");
    }

    string port;
    pid_t pid;
    int status;
    if (ServerStart(timeout, serverPath, pid, port, logLevel) != 0) {
        LOGE("start server fail\n");
        return 0;
    }
    ClientStart(timeout, arg, pluginName, port);
    while (1) {
        if (PluginClient::GetInstance()->GetInjectFlag()) {
            register_callback(pluginName.c_str(), PLUGIN_FINISH, &GccEnd, NULL);
            break;
        }
        if (pid == waitpid(-1, &status, WNOHANG)) {
            break;
        }
    }
    return 0;
}

