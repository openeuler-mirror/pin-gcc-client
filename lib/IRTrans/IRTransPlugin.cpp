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
#include "IRTrans/IRTransPlugin.h"
#include "context.h"
#include "tree-pass.h"
#include "tree.h"

using namespace PinClient;

using std::vector;
int plugin_is_GPL_compatible;
static pid_t g_serverPid;

/* gcc插件end事件回调函数 */
static void GccEnd(void *gccData, void *userData)
{
    int status = 0;
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    if (client == nullptr) {
        return;
    }
    LOGI("gcc optimize has been done! now close server...\n");
    client->ReceiveSendMsg("stop", "");
    if (client->GetUserFuncState() != STATE_TIMEOUT) {
        waitpid(g_serverPid, &status, 0);
    } else {
        client->DeletePortFromLockFile(client->GetGrpcPort());
    }

    LOGI("client pid:%d quit\n", getpid());
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
    LOGI("%s end!\n", __func__);
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
    HANDLE_MANAGER_SETUP,
    HANDLE_MAX
};

int RegisterPluginEvent(InjectPoint inject, const string& pluginName)
{
    plugin_event event;
    if (PluginClient::GetEvent(inject, &event) != 0) {
        return -1;
    }
    LOGD("%s inject:%d,%s\n", __func__, inject, pluginName.c_str());
    register_callback(pluginName.c_str(), event, &GccEventCallback, (void *)&g_event[inject]);
    return 0;
}

void ManagerSetupCallback(void)
{
    string key = "injectPoint";
    InjectPoint inject = HANDLE_MANAGER_SETUP;
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    vector<string> userFuncs = client->GetFuncNameByInject(inject);
    for (auto &userFunc : userFuncs) {
        if (client->GetUserFuncState() == STATE_TIMEOUT) {
            break;
        }
        string value = std::to_string(inject) + ":" + userFunc;
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

struct RltPass: rtl_opt_pass {
public:
    RltPass(pass_data passData): rtl_opt_pass(passData, g)
    {
    }
    unsigned int execute(function *fun) override
    {
        ManagerSetupCallback();
        return 0;
    }
    RltPass* clone() override
    {
        return this;
    }
};

struct SimpleIPAPass: simple_ipa_opt_pass {
public:
    SimpleIPAPass(pass_data passData): simple_ipa_opt_pass(passData, g)
    {
    }
    unsigned int execute(function *fun) override
    {
        ManagerSetupCallback();
        return 0;
    }
    SimpleIPAPass* clone() override
    {
        return this;
    }
};

struct GimplePass: gimple_opt_pass {
public:
    GimplePass(pass_data passData): gimple_opt_pass(passData, g)
    {
    }
    unsigned int execute(function *fun) override
    {
        ManagerSetupCallback();
        return 0;
    }
    GimplePass* clone() override
    {
        return this;
    }
};

static std::map<RefPassName, string> g_refPassName {
    {PASS_CFG, "cfg"},
    {PASS_PHIOPT, "phiopt"},
    {PASS_SSA, "ssa"},
    {PASS_LOOP, "loop"},
};

int RegisterPassManagerSetup(InjectPoint inject, const ManagerSetupData& setupData, const string& pluginName)
{
    if (inject != HANDLE_MANAGER_SETUP) {
        return -1;
    }

    struct register_pass_info passInfo;
    pass_data passData = {
        .type = GIMPLE_PASS,
        .name = "managerSetupPass",
        .optinfo_flags = OPTGROUP_NONE,
        .tv_id = TV_NONE,
        .properties_required = 0,
        .properties_provided = 0,
        .properties_destroyed = 0,
        .todo_flags_start = 0,
        .todo_flags_finish = 0,
    };

    passInfo.reference_pass_name = g_refPassName[setupData.refPassName].c_str();
    passInfo.ref_pass_instance_number = setupData.passNum;
    passInfo.pos_op = (pass_positioning_ops)setupData.passPosition;
    switch (setupData.refPassName) {
        case PASS_CFG:
            passData.type = GIMPLE_PASS;
            passInfo.pass = new GimplePass(passData);
            break;
        case PASS_PHIOPT:
            passData.type = GIMPLE_PASS;
            passInfo.pass = new GimplePass(passData);
            break;
        case PASS_SSA:
            passData.type = RTL_PASS;
            passInfo.pass = new RltPass(passData);
            break;
        case PASS_LOOP:
            passData.type = SIMPLE_IPA_PASS;
            passInfo.pass = new SimpleIPAPass(passData);
            break;
        default:
            passInfo.pass = new GimplePass(passData);
            break;
    }
    
    register_callback(pluginName.c_str(), PLUGIN_PASS_MANAGER_SETUP, NULL, &passInfo);
    return 0;
}

static bool PluginVersionCheck(struct plugin_gcc_version *gccVersion, struct plugin_gcc_version *pluginVersion)
{
    if (!gccVersion || !pluginVersion) {
        return false;
    }

    string gccVer = gccVersion->basever;
    string pluginVer = pluginVersion->basever;
    string gccVerMajor = gccVer.substr(0, gccVer.find_first_of(".", gccVer.find_first_of(".") + 1));
    string pluginVerMajor = pluginVer.substr(0, pluginVer.find_first_of(".", pluginVer.find_first_of(".") + 1));
    if (gccVerMajor != pluginVerMajor) {
        return false;
    }
    return true;
}

int plugin_init(struct plugin_name_args *pluginInfo, struct plugin_gcc_version *version)
{
    string pluginName = pluginInfo->base_name;
    register_callback(pluginName.c_str(), PLUGIN_FINISH, &GccEnd, NULL);

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
    int status;
    if (ServerStart(timeout, serverPath, g_serverPid, port, logLevel) != 0) {
        LOGE("start server fail\n");
        return 0;
    }
    ClientStart(timeout, arg, pluginName, port);
    std::shared_ptr<PluginClient> client = PluginClient::GetInstance();
    while (1) {
        if ((client->GetInjectFlag()) || (client->GetUserFuncState() == STATE_TIMEOUT)) {
            break;
        }
        if (g_serverPid == waitpid(-1, &status, WNOHANG)) {
            PluginClient::DeletePortFromLockFile((unsigned short)atoi(port.c_str()));
            break;
        }
    }
    return 0;
}

