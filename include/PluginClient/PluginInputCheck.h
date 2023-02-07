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
    This file contains the declaration of the PluginInputCheck class.
    主要完成功能：获取gcc编译时的参数，读取config.json配置文件，完成sha256校验，完成
    安全编译选项参数检查，并保存serverPath，shaPath，timeout等信息
*/

#ifndef PLUGIN_INPUT_CHECK_H
#define PLUGIN_INPUT_CHECK_H

#include <map>
#include <string>
#include <json/json.h>
#include "gcc-plugin.h"
#include "PluginClient/PluginLog.h"

namespace PinClient {
using std::map;
using std::string;
using std::vector;

enum Jsonkey {
#define JSON_KEY(KEY, NAME) KEY,
#include "JsonKey.def"
#undef JSON_KEY
MAX_JSON_KEYS
};

#define JSON_KEY(KEY, NAME) NAME,
const char *const jsonkey[] = {
#include "JsonKey.def"
};
#undef JSON_KEY

enum {
    KEY_SERVER_PATH,
    KEY_LOG_LEVEL,
};

class PluginInputCheck {
public:
    const string shaFile = "/libpin_user.sha256";
    PluginInputCheck();
    /* 从配置文件读取初始化信息 */
    int GetInitInfo();
    int CheckServerFile(); // 对server文件进行检查
    int CheckShaFile(); // 对sha256校验文件进行检查
    bool ReadConfigfile(Json::Value& root);
    /* 进行sha256校验 */
    int CheckSHA256();
    void CheckSafeCompileFlag(const string& argName, const string& param);
    /* 解析gcc编译时传递的-fplugin-arg参数 */
    void GetInputArgs(struct plugin_name_args *pluginInfo);
    int mapFind(const std::map<string, int>& map, const string& key)
    {
        auto it = map.find(key);
        if (it != map.end()) {
            return it->second;
        }
        return -1;
    }
    string& GetArgs()
    {
        return args;
    }
    string& GetServerPath()
    {
        return serverPath;
    }
    void SetServerPath(const string& path)
    {
        serverPath = path;
    }
    LogPriority GetLogLevel()
    {
        return logLevel;
    }
    void SetLogLevel(LogPriority level)
    {
        logLevel = level;
    }
    string& GetShaPath()
    {
        return shaPath;
    }
    void SetShaPath(const string& path)
    {
        shaPath = path;
    }
    int GetTimeout()
    {
        return timeout;
    }
    void SetTimeout(int time);
    void SetConfigPath(const string& path)
    {
        configFilePath = path;
    }
    string& GetConfigPath()
    {
        return configFilePath;
    }
    string GetServerDir()
    {
        int index = serverPath.find_last_of("/");
        return serverPath.substr(0, index);
    }

private:
    string args;
    string serverPath;
    string configFilePath;
    LogPriority logLevel;
    string shaPath;
    int timeout;
    vector<string> safeCompileFlags;
};
} // namespace PinClient
#endif