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
//===----------------------------------------------------------------------===//
// 
// This is the header file for the Plugin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PLUGIN_DIALECT_H
#define PLUGIN_DIALECT_H

#include "mlir/IR/Dialect.h"

/// Include the patterns defined in the Declarative Rewrite framework.
#include "Dialect/PluginOpsDialect.h.inc"

#endif // PLUGIN_DIALECT_H
