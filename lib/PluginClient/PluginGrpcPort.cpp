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
    This file contains the implementation of the PluginGrpcPort class.
    主要完成功能：查找未使用的端口号，并将端口号写入到共享文件中，通过文件锁控制多进程间访问，并提供
    DeletePortFromLockFile接口在退出时删除写入的端口号
*/

#include <fstream>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include "PluginClient/PluginLog.h"
#include "PluginClient/PluginGrpcPort.h"

namespace PinClient {
int PluginGrpcPort::OpenFile(const char *path)
{
    int portFileFd = -1;
    if (access(path, F_OK) == -1) {
        mode_t mask = umask(0);
        mode_t mode = 0666; // 权限是rwrwrw，跨进程时，其他用户也要可以访问
        portFileFd = open(path, O_CREAT | O_RDWR, mode);
        umask(mask);
    } else {
        portFileFd = open(path, O_RDWR);
    }

    if (portFileFd == -1) {
        LOGE("open file %s fail\n", path);
    }
    return portFileFd;
}

bool PluginGrpcPort::ReadPortsFromLockFile(int fd, string& grpcPorts)
{
    if (flock(fd, LOCK_EX) != 0) {
        return false;
    }

    int fileLen = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    char *buf = new char[fileLen + 1];
    if (buf == NULL) {
        return false;
    }
    if (read(fd, buf, fileLen) < 0) {
        return false;
    }
    buf[fileLen] = '\0';
    grpcPorts = buf;
    delete[] buf;
    return true;
}

bool PluginGrpcPort::FindUnusedPort()
{
    unsigned short basePort = GetBasePort();
    int portFileFd = OpenFile(lockFilePath.c_str());
    if (portFileFd == -1) {
        return false;
    }

    string grpcPorts = "";
    if (!ReadPortsFromLockFile(portFileFd, grpcPorts)) {
        close(portFileFd);
        return false;
    }

    // 不使用UINT16_MAX端口号，端口号为UINT16_MAX时作异常处理
    while (++basePort < UINT16_MAX) {
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        struct sockaddr_in serverAddr;
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = inet_addr("0.0.0.0");
        serverAddr.sin_port = htons(basePort);
        int ret = connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
        if (sock != -1) {
            close(sock);
        }
        if ((ret == -1) && (errno == ECONNREFUSED)) {
            string strPort = std::to_string(basePort) + "\n";
            if (grpcPorts.find(strPort) == grpcPorts.npos) {
                port = basePort;
                LOGI("found port:%d\n", port);
                lseek(portFileFd, 0, SEEK_END);
                write(portFileFd, strPort.c_str(), strPort.size());
                break;
            }
        }
    }

    if (basePort == UINT16_MAX) {
        ftruncate(portFileFd, 0); // 清空锁文件，避免异常未删除释放的端口号，导致无端口使用
        lseek(portFileFd, 0, SEEK_SET);
        close(portFileFd); // 关闭文件fd会同时释放文件锁
        return false;
    }

    close(portFileFd); // 关闭文件fd会同时释放文件锁
    return true;
}

bool PluginGrpcPort::DeletePortFromLockFile()
{
    if (port == 0) {
        return true;
    }
    int portFileFd = open(lockFilePath.c_str(), O_RDWR);
    if (portFileFd == -1) {
        LOGE("%s open file %s fail\n", __func__, lockFilePath.c_str());
        return false;
    }
    LOGI("delete port:%d\n", port);

    string grpcPorts = "";
    if (!ReadPortsFromLockFile(portFileFd, grpcPorts)) {
        close(portFileFd);
        port = 0;
        return false;
    }

    string portStr = std::to_string(port) + "\n";
    string::size_type pos = grpcPorts.find(portStr);
    if (pos == string::npos) {
        close(portFileFd);
        port = 0;
        return true;
    }
    grpcPorts = grpcPorts.erase(pos, portStr.size());

    ftruncate(portFileFd, 0);
    lseek(portFileFd, 0, SEEK_SET);
    write(portFileFd, grpcPorts.c_str(), grpcPorts.size());
    close(portFileFd);
    port = 0;
    return true;
}
} // namespace PinClient
