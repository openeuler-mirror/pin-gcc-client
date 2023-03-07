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
    This file contains the implementation of the PluginInputCheck class.
*/

#include <fstream>
#include "PluginClient/PluginInputCheck.h"

namespace PinClient {
// 对server可执行文件进行检查,后续可以在此函数中扩展是否可执行，文件权限等检查
int PluginInputCheck::CheckServerFile()
{
    int ret = access(serverPath.c_str(), F_OK);
    return ret;
}

// 对sha256文件进行检查,后续可以在此函数中扩展文件权限、是否为空等检查
int PluginInputCheck::CheckShaFile()
{
    int ret = access(shaPath.c_str(), F_OK);
    return ret;
}

bool PluginInputCheck::ReadConfigfile(Json::Value& root)
{
    Json::Reader reader;
    std::ifstream ifs(configFilePath.c_str());
    if (!ifs.is_open()) {
        return false;
    }

    if (!reader.parse(ifs, root)) {
        fprintf(stderr, "parse %s fail! check the file format!\n", configFilePath.c_str());
        ifs.close();
        return false;
    }
    
    ifs.close();
    return true;
}

void PluginInputCheck::SetTimeout(int time)
{
    const int timeoutMin = 50;
    const int timeoutMax = 5000;
    if ((time >= timeoutMin) && (time <= timeoutMax)) { // 不在50~5000ms范围内，使用默认值
        timeout = time;
        LOGI("the timeout is:%d\n", timeout);
        return;
    }
    LOGW("SetTimeout:%d,should be 50~5000,use default:%d\n", time, timeout);
}

int PluginInputCheck::GetInitInfo()
{
    Json::Value root;
    if (!ReadConfigfile(root)) {
        return -1;
    }

    if (serverPath == "") {
        if (root[jsonkey[PATH]].isString()) {
            serverPath = root[jsonkey[PATH]].asString();
        } else {
            LOGW("serverPath in config.json is not string!\n");
        }
    }
    if (CheckServerFile() != 0) {
        LOGE("serverPath:%s not exist!\n", serverPath.c_str());
        serverPath = "";
        return -1;
    }

    if (root[jsonkey[TIMEOUT]].isInt()) {
        int timeoutJson = root[jsonkey[TIMEOUT]].asInt();
        SetTimeout(timeoutJson);
    } else {
        LOGW("timeout in config.json is not int or out of int range!use default:%d\n", timeout);
    }

    if (root[jsonkey[SHA256]].isString()) {
        shaPath = root[jsonkey[SHA256]].asString();
    } else {
        LOGW("sha256file in config.json is not string!\n");
    }

    if ((shaPath == "") || (CheckShaFile() != 0)) {
        shaPath = GetServerDir() + shaFile; // sha256文件默认和server在同一目录
        LOGW("sha256 file not found,use default:%s\n", shaPath.c_str());
    }
    return 0;
}

std::map<string, int> g_keyMap {
    {"server_path", KEY_SERVER_PATH},
    {"log_level", KEY_LOG_LEVEL},
};
void PluginInputCheck::GetInputArgs(struct plugin_name_args *pluginInfo)
{
    Json::Value root;
    map<string, string> compileArgs;

    for (int i = 0; i < pluginInfo->argc; i++) {
        string key = pluginInfo->argv[i].key;
        int keyTag = -1;
        auto it = g_keyMap.find(key);
        if (it != g_keyMap.end()) {
            keyTag = it->second;
        }
        switch (keyTag) {
            case KEY_SERVER_PATH:
                serverPath = pluginInfo->argv[i].value;
                if (serverPath != "") {
                    configFilePath = GetServerDir() + "/pin-gcc-client.json"; // 配置文件和server在同一目录
                    shaPath = GetServerDir() + shaFile; // sha256文件默认和server在同一目录
                }
                break;
            case KEY_LOG_LEVEL:
                logLevel = (LogPriority)atoi(pluginInfo->argv[i].value);
                SetLogPriority(logLevel);
                break;
            default:
                string value = pluginInfo->argv[i].value;
                compileArgs[key] = value;
                root[key] = value;
                break;
        }
    }

    args = root.toStyledString();
    for (auto it = compileArgs.begin(); it != compileArgs.end(); it++) {
        CheckSafeCompileFlag(it->first, it->second);
    }
}

int PluginInputCheck::CheckSHA256()
{
    if (shaPath == "") {
        LOGE("sha256file Path is NULL!\n");
        return -1;
    }
    int index = shaPath.find_last_of("/");
    string dir = shaPath.substr(0, index);
    string filename = shaPath.substr(index+1, -1);

    string cmd = "cd " + dir + " && " + "sha256sum -c " + filename + " --quiet";
    int ret = system(cmd.c_str());
    return ret;
}

void PluginInputCheck::CheckSafeCompileFlag(const string& argName, const string& param)
{
    for (auto& compileFlag : safeCompileFlags) {
        if (param.find(compileFlag) != string::npos) {
            LOGW("%s:%s have safe compile parameter:%s !!!\n", argName.c_str(), param.c_str(), compileFlag.c_str());
        }
    }
}

PluginInputCheck::PluginInputCheck()
{
    shaPath = "";
    serverPath = "";
    logLevel = PRIORITY_WARN;
    timeout = 200; // 默认超时时间200ms
    SetConfigPath("/usr/local/bin/pin-gcc-client.json");
    safeCompileFlags.push_back("-z noexecstack");
    safeCompileFlags.push_back("-fno-stack-protector");
    safeCompileFlags.push_back("-fstack-protector-all");
    safeCompileFlags.push_back("-D_FORTIFY_SOURCE");
    safeCompileFlags.push_back("-fPic");
    safeCompileFlags.push_back("-fPIE");
    safeCompileFlags.push_back("-fstack-protector-strong");
    safeCompileFlags.push_back("-fvisibility");
    safeCompileFlags.push_back("-ftrapv");
    safeCompileFlags.push_back("-fstack-check");
}
} // namespace PinClient
