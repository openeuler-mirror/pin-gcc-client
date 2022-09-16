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
    This file contains the declaration of the BasicPluginAPI class.
*/

#ifndef BASIC_PLUGIN_FRAMEWORK_API_H
#define BASIC_PLUGIN_FRAMEWORK_API_H

#include <vector>
#include <string>
#include "IR/Operation.h"
#include "IR/Decl.h"

namespace Plugin_API {
using namespace Plugin_IR;
using std::vector;
using std::string;

/* The BasicPluginAPI class defines the basic plugin API, both the plugin
   client and the server should inherit this class and implement there own
   defined API. */
class BasicPluginAPI {
public:
    BasicPluginAPI() = default;
    virtual ~BasicPluginAPI() = default;

    virtual vector<Operation> SelectOperation(Opcode op, string attribute) = 0;
    virtual vector<Operation> GetAllFunc(string attribute) = 0;
    virtual Decl SelectDeclByID(uintptr_t id) = 0;
}; // class BasicPluginAPI
} // namespace Plugin_API

#endif // PLUGIN_FRAMEWORK_API_H