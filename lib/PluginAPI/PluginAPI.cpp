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