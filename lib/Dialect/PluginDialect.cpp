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
// This file defines Plugin dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/PluginDialect.h"
#include "Dialect/PluginOps.h"

using namespace mlir;
using namespace mlir::Plugin;

//===----------------------------------------------------------------------===//
// Plugin dialect.
//===----------------------------------------------------------------------===//

void PluginDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/PluginOps.cpp.inc"
      >();
}