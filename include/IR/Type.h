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
   Description: This file contains the declaration of the type class.
*/

#ifndef PLUGIN_IR_TYPE_H
#define PLUGIN_IR_TYPE_H

#include <string>
#include <map>

namespace Plugin_IR {
using std::map;
using std::string;

/* Enum of type code */
enum TypeCode : uint8_t {
    TC_UNDEF,
#define DEF_CODE(NAME, TYPE) TC_##NAME,
#include "TypeCode.def"
#undef DEF_CODE
    TC_END
};

/* Enum of type qualifiers */
enum TypeQualifiers : uint8_t {
    TQ_UNDEF = 1 << 0,
    TQ_CONST = 1 << 1,
    TQ_VOLATILE = 1 << 2,
    TQ_END = TQ_CONST | TQ_VOLATILE,
};

/* The type class defines the type of plugin IR. */
class Type {
public:
    Type () = default;
    ~Type () = default;

    inline void SetID(uintptr_t id)
    {
        this->id = id;
    }

    inline uintptr_t GetID() const
    {
        return id;
    }

    inline void SetTypeCode(TypeCode op)
    {
        typeCode = op;
    }

    inline TypeCode GetTypeCode() const
    {
        return typeCode;
    }

    inline void SetTQual(uint8_t op)
    {
        tQual = op;
    }

    inline uint8_t GetTQual() const
    {
        return tQual;
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
    TypeCode typeCode;
    uint8_t tQual;
    map<string, string> attributes;
}; // class Type
} // namespace Plugin_IR

#endif // PLUGIN_IR_TYPE_H