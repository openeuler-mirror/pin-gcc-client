/*
   Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may
   not use this file except in compliance with the License. You may obtain
   a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
   License for the specific language governing permissions and limitations
   under the License.

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