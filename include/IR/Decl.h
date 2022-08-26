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
    This file contains the declaration of the decl class.
*/

#ifndef PLUGIN_IR_DECL_H
#define PLUGIN_IR_DECL_H

#include <string>
#include <map>
#include "IR/Type.h"

namespace Plugin_IR {
using std::map;
using std::string;

/* Enum of decl code */
enum DeclCode : uint8_t {
    DC_UNDEF,
#define DEF_CODE(NAME, TYPE) DC_##NAME,
#include "DeclCode.def"
#undef DEF_CODE
    DC_END
};

/* The decl class defines the decl of plugin IR. */
class Decl {
public:
    Decl () = default;
    ~Decl () = default;

    Decl (DeclCode op)
    {
        declCode = op;
    }

    inline void SetID(uintptr_t id)
    {
        this->id = id;
    }

    inline uintptr_t GetID() const
    {
        return id;
    }

    inline void SetDeclCode(DeclCode op)
    {
        declCode = op;
    }

    inline DeclCode GetDeclCode() const
    {
        return declCode;
    }

    inline void SetType(Type t)
    {
        declType = t;
    }

    inline Type GetType() const
    {
        return declType;
    }

    bool AddAttribute(string key, string val, bool force = false)
    {
        if (!force) {
            if (attributes.find(key) != attributes.end()) {
                return false;
            }
        }
        attributes[key] = val;
        return true;
    }

    string GetAttribute(string key) const
    {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            return it->second;
        }
        return "";
    }

    map<string, string>& GetAttributes()
    {
        return attributes;
    }

private:
    uintptr_t id;
    DeclCode declCode;
    map<string, string> attributes;
    Type declType;
}; // class Decl
} // namespace Plugin_IR

#endif // PLUGIN_IR_DECL_H