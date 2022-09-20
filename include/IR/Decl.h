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