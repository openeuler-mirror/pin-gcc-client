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
    This file contains the functions of gcc plugin callback and init.
    主要完成功能：实现RegisterPluginEvent和RegisterPassManagerSetup功能，完成gcc
    版本号检查，当触发注册点对应事件时，回调server注册的用户函数
    plugin_init为整个程序入口函数
*/

#include <string>
#include "PluginClient/PluginClient.h"
#include "plugin-version.h"
#include "tree-pass.h"
#include "tree.h"
#include "context.h"
#include "gccPlugin/gccPlugin.h"

using namespace PinClient;

int plugin_is_GPL_compatible;
static pid_t g_serverPid;

/* gcc插件end事件回调函数,停止插件服务端，等待子进程结束，删除端口号 */
void GccEnd(void *gccData, void *userData)
{
    int status = 0;
    PluginClient *client = PluginClient::GetInstance();
    if (!client->GetStartFlag()) {
        return;
    }

    LOGI("gcc optimize has been done! now close server...\n");
    client->ReceiveSendMsg("stop", "");
    if (client->GetUserFuncState() != STATE_TIMEOUT) {
        waitpid(g_serverPid, &status, 0);
    } else {
        client->DeleteGrpcPort();
    }
    LOGI("client pid:%d quit\n", getpid());
}

static void WaitIRTrans(void *gccData, PluginClient *client)
{
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
                client->GetIRTransResult(gccData, funcName, param);
            }
        }
    }
}

/* gcc插件回调函数,当注册的plugin_event触发时,进入此函数 */
void GccEventCallback(void *gccData, void *userData)
{
    PluginClient *client = PluginClient::GetInstance();
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
        WaitIRTrans(gccData, client);
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
    HANDLE_INCLUDE_FILE,
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

static void ManagerSetupCallback(unsigned int index, function *fun)
{
    string key = "injectPoint";
    InjectPoint inject = HANDLE_MANAGER_SETUP;
    PluginClient *client = PluginClient::GetInstance();
    vector<string> userFuncs = client->GetFuncNameByInject(inject);
    if (index < userFuncs.size()) {
        string name = userFuncs[index].substr(0, userFuncs[index].find_first_of(","));
        string value = std::to_string(inject) + ":" + name + ",params:" + std::to_string((uintptr_t)fun);
        client->ReceiveSendMsg(key, value);
        WaitIRTrans(nullptr, client);
    }
}

struct RtlPass: rtl_opt_pass {
public:
    RtlPass(pass_data passData, unsigned int indx): rtl_opt_pass(passData, g), index(indx)
    {
    }
    /* unsigned int execute(function *fun) override
    {
        ManagerSetupCallback(index, fun);
        return 0;
    } */

private:
    unsigned int index;
};

struct SimpleIPAPass: simple_ipa_opt_pass {
public:
    SimpleIPAPass(pass_data passData, unsigned int indx): simple_ipa_opt_pass(passData, g), index(indx)
    {
    }
    /* unsigned int execute(function *fun) override
    {
        ManagerSetupCallback(index, fun);
        return 0;
    } */

private:
    unsigned int index;
};

struct GimplePass: gimple_opt_pass {
public:
    GimplePass(pass_data passData, unsigned int indx): gimple_opt_pass(passData, g), index(indx)
    {
    }
    unsigned int execute(function *fun) override
    {
        ManagerSetupCallback(index, fun);
        return 0;
    }
    GimplePass* clone() override
    {
        return this;
    }

private:
    unsigned int index;
};

static std::map<RefPassName, string> g_refPassName {
    {PASS_CFG, "cfg"},
    {PASS_PHIOPT, "phiopt"},
    {PASS_SSA, "ssa"},
    {PASS_LOOP, "loop"},
};

void RegisterPassManagerSetup(unsigned int index, const ManagerSetupData& setupData, const string& pluginName)
{
    struct register_pass_info passInfo;
    string passDataName = "managerSetupPass_" + g_refPassName[setupData.refPassName];
    pass_data passData = {
        .type = GIMPLE_PASS,
        .name = passDataName.c_str(),
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
            passInfo.pass = new GimplePass(passData, index);
            break;
        case PASS_PHIOPT:
            passData.type = GIMPLE_PASS;
            passInfo.pass = new GimplePass(passData, index);
            break;
        case PASS_SSA:
            passData.type = GIMPLE_PASS;
            passInfo.pass = new GimplePass(passData, index);
            break;
        case PASS_LOOP:
            passData.type = GIMPLE_PASS;
            passInfo.pass = new GimplePass(passData, index);
            break;
        default:
            passInfo.pass = new GimplePass(passData, index);
            break;
    }
    
    if (pluginName != "") {
        register_callback(pluginName.c_str(), PLUGIN_PASS_MANAGER_SETUP, NULL, &passInfo);
    }
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
    if (!PluginVersionCheck(version, &gcc_version)) {
        LOGE("incompatible gcc/plugin versions\n");
        return 1;
    }
    string pluginName = pluginInfo->base_name;
    register_callback(pluginName.c_str(), PLUGIN_FINISH, &GccEnd, NULL);

    PluginClient::GetInstance()->Init(pluginInfo, pluginName, g_serverPid);
    return 0;
}

