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