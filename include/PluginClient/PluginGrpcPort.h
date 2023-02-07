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
    This file contains the declaration of the PluginGrpcPort class.
    主要完成功能：查找未使用的端口号，并将端口号写入到共享文件中，通过文件锁控制多进程间访问，并提供
    DeletePortFromLockFile接口在退出时删除写入的端口号
*/

#ifndef PLUGIN_GRPC_PORT_H
#define PLUGIN_GRPC_PORT_H

#include <string>

namespace PinClient {
using std::string;

class PluginGrpcPort {
public:
    PluginGrpcPort()
    {
        const int startPort = 40000;
        lockFilePath = "/tmp/grpc_ports_pin_client.txt";
        basePort = startPort;
        port = 0;
    }
    bool FindUnusedPort(); // 查找未被使用的端口号，确保并发情况下server和client一对一
    /* grpc的server被client拉起之前将port记录在/tmp/grpc_ports_pin_client.txt中, server和client建立通信后从文件中删除port，避免多进程时端口冲突
       文件若不存在，先创建文件 */
    int OpenFile(const char *path);
    /* 读取文件中保存的grpc端口号 */
    bool ReadPortsFromLockFile(int fd, string& grpcPorts);
    /* server启动异常或者grpc建立通信后,将文件中记录的端口号删除 */
    bool DeletePortFromLockFile();
    unsigned short GetPort()
    {
        return port;
    }
    void SetLockFilePath(const string& path)
    {
        lockFilePath = path;
    }
    void SetBasePort(unsigned short port)
    {
        basePort = port;
    }
    unsigned short GetBasePort()
    {
        return basePort;
    }

private:
    unsigned short port; // server和client使用的端口号
    string lockFilePath;
    unsigned short basePort;
};
} // namespace PinClient
#endif