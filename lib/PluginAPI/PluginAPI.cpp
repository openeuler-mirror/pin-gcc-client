/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

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
    This file contains the implementation of the client PluginAPI class.
*/

#include "PluginAPI/PluginAPI_Client.h"

namespace Plugin_API {
vector<Operation> PluginAPI_Client::SelectOperation(Opcode op, string attribute)
{
    vector<Operation> retOps;
    if (op == OP_FUNCTION) {
        vector<Operation> allFunction = irTrans.GetAllFunction();
        if (allFunction.empty())
            return retOps;
        if (attribute.empty())
            return allFunction;
        for (auto& f : allFunction)
            if (f.GetAttribute(attribute) == "1")
                retOps.push_back(f);
    }
    return retOps;
}

vector<Operation> PluginAPI_Client::GetAllFunc(string attribute)
{
    vector<Operation> retOps;
        vector<Operation> allFunction = irTrans.GetAllFunction();
        if (allFunction.empty()) return retOps;
        for (auto& f : allFunction)
            if (f.GetAttribute(attribute) != "") {
                retOps.push_back(f);
            }
    return retOps;
}

Decl PluginAPI_Client::SelectDeclByID(uintptr_t id)
{
    return irTrans.GetDeclByID(id);
}
} // namespace Plugin_API