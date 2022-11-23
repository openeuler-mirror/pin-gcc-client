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

*/

#ifndef PLUGIN_OPS_CLIENT_API_H
#define PLUGIN_OPS_CLIENT_API_H

#include "BasicPluginOpsAPI.h"
#include "Translate/GimpleToPluginOps.h"

namespace PluginAPI {
using std::vector;
using std::string;
using namespace mlir::Plugin;

class PluginClientAPI : public BasicPluginOpsAPI {
public:
    PluginClientAPI (mlir::MLIRContext &context) : gimpleConversion(context) {};
    PluginClientAPI () = default;
    ~PluginClientAPI () = default;

    vector<FunctionOp> GetAllFunc() override;

private:
    PluginIR::GimpleToPluginOps gimpleConversion;
}; // class PluginClientAPI
} // namespace PluginAPI

#endif // PLUGIN_OPS_CLIENT_API_H