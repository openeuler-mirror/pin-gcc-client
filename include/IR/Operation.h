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
    This file contains the declaration of the Operation class.
*/

#ifndef PLUGIN_IR_OPERATION_H
#define PLUGIN_IR_OPERATION_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "IR/Decl.h"
#include "IR/Type.h"

namespace Plugin_IR {
using std::vector;
using std::map;
using std::string;
using std::shared_ptr;

/* Enum of opcode */
enum Opcode : uint8_t {
    OP_UNDEF,
#define DEF_CODE(NAME, TYPE) OP_##NAME,
#include "OperationCode.def"
#undef DEF_CODE
    OP_END
};

/* The operation defines the operation of plugin IR. */
class Operation {
public:
    Operation () = default;
    ~Operation () = default;

    Operation (Opcode op)
    {
        opcode = op;
    }

    inline void SetID(uintptr_t id)
    {
        this->id = id;
    }

    inline uintptr_t GetID() const
    {
        return id;
    }

    inline void SetOpcode(Opcode op)
    {
        opcode = op;
    }

    inline Opcode GetOpcode() const
    {
        return opcode;
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

    bool AddOperand(string key, Decl val, bool force = false)
    {
        if (!force) {
            if (operands.find(key) != operands.end()) {
                return false;
            }
        }
        operands[key] = val;
        return true;
    }

    map<string, string>& GetAttributes()
    {
        return attributes;
    }
    
    Type& GetResultTypes()
    {
        return resultType;
    }
    
    map<string, Decl>& GetOperands()
    {
        return operands;
    }

    bool AddSuccessor(shared_ptr<Operation> succ)
    {
        successors.push_back(succ);
        return true;
    }

    vector<shared_ptr<Operation>>& GetSuccessors()
    {
        return successors;
    }

    void Dump()
    {
        printf ("operation: {");
        switch (opcode) {
            case OP_FUNCTION:
                printf(" opcode: OP_FUNCTION\n");
                break;
            default:
                printf(" opcode: unhandled\n");
                break;
        }
        if (!attributes.empty()) {
            printf (" attributes:\n");
            for (const auto& attr : attributes) {
                printf ("    %s:%s\n", attr.first.c_str(), attr.second.c_str());
            }
        }
        printf ("}\n");
    }

private:
    uintptr_t id;
    Opcode opcode;
    Type resultType;
    vector<shared_ptr<Operation>> successors;
    vector<shared_ptr<Operation>> regions;
    map<string, Decl> operands;
    map<string, string> attributes;
}; // class Operation
} // namespace Plugin_IR

#endif // PLUGIN_IR_OPERATION_H