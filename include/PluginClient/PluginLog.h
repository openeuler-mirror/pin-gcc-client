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
    This file contains the declaration of the Plugin_Log class.
*/

#ifndef PLUGIN_LOG_H
#define PLUGIN_LOG_H

namespace PinClient {
#define LOG_FILE_SIZE   (10 * 1024 * 1024)

enum LogPriority : uint8_t {
    PRIORITY_ERROR = 0,
    PRIORITY_WARN,
    PRIORITY_INFO,
    PRIORITY_DEBUG
};
void LogPrint(LogPriority priority, const char *tag, const char *fmt, ...);
void CloseLog(void);
bool SetLogPriority(LogPriority priority);

#define LOGE(...) LogPrint(PRIORITY_ERROR, "ERROR:", __VA_ARGS__)
#define LOGW(...) LogPrint(PRIORITY_WARN, "WARN:", __VA_ARGS__)
#define LOGI(...) LogPrint(PRIORITY_INFO, "", __VA_ARGS__)
#define LOGD(...) LogPrint(PRIORITY_DEBUG, "DEBUG:", __VA_ARGS__)
} // namespace PinClient

#endif
