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