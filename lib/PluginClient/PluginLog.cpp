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
    This file contains the implementation of the Plugin_Log class
*/

#include <cstdarg>
#include <iostream>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <csignal>
#include "PluginClient/PluginLog.h"

namespace PinClient {
using namespace std;
using std::string;
constexpr int LOG_BUF_SIZE = 10240;
constexpr int BASE_DATE = 1900;
static LogPriority g_priority = PRIORITY_WARN; // log打印的级别控制
static std::mutex g_mutex; // 线程锁

shared_ptr<fstream> g_fs;
static void LogWriteInit(const string& data);
static void (*g_writeToLog)(const string& data) = LogWriteInit;

static void GetLogFileName(string& fileName)
{
    time_t nowTime = time(nullptr);
    struct tm *t = localtime(&nowTime);
    char buf[100];
    sprintf(buf, "/tmp/pin_client%d_%4d%02d%02d_%02d_%02d_%02d.log", getpid(),
        t->tm_year + BASE_DATE, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fileName = buf;
}

static void LogWriteFile(const string& data)
{
    if (g_fs->tellg() > LOG_FILE_SIZE) {
        g_fs->close();
        string fileName;
        GetLogFileName(fileName);
        g_fs->open(fileName.c_str(), ios::app);
    }

    g_fs->write(data.c_str(), data.size());
}

static void LogWriteInit(const string& data)
{
    if (g_writeToLog == LogWriteInit) {
        g_fs = std::make_shared<fstream>();
        string fileName;
        GetLogFileName(fileName);
        g_fs->open(fileName.c_str(), ios::app);
        g_writeToLog = LogWriteFile;
    }
    g_writeToLog(data);
}

void CloseLog(void)
{
    if (g_fs) {
        if (g_fs->is_open()) {
            g_fs->close();
        }
    }
}

static void LogWrite(const char *tag, const char *msg)
{
    time_t nowTime = time(nullptr);
    struct tm *t = localtime(&nowTime);
    char buf[30];
    sprintf(buf, "%4d-%02d-%02d %02d:%02d:%02d ", t->tm_year + BASE_DATE, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);

    string stag = tag;
    string smsg = msg;
    string data = buf + stag + smsg;
    g_writeToLog(data);
}

void LogPrint(LogPriority priority, const char *tag, const char *fmt, ...)
{
    va_list ap;
    char buf[LOG_BUF_SIZE];

    va_start(ap, fmt);
    vsnprintf(buf, LOG_BUF_SIZE, fmt, ap);
    va_end(ap);

    if (priority <= g_priority) {
        printf("%s%s", tag, buf);
    }

    g_mutex.lock();
    LogWrite(tag, buf);
    g_mutex.unlock();
}

bool SetLogPriority(LogPriority priority)
{
    if (priority > PRIORITY_DEBUG) {
        return false;
    }
    g_priority = priority;
    return true;
}
} // namespace PinClient
