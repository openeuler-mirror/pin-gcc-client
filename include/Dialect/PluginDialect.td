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
// This is the top level file for the plugin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PLUGIN_DIALECT_TD
#define PLUGIN_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Plugin Dialect Definition.
//===----------------------------------------------------------------------===//

// Provide a definition of the 'Plugin' dialect in the ODS framework so that we
// can define our operations.
def Plugin_Dialect : Dialect {
  let name = "Plugin";
  let summary = "The Plugin Dialect.";
  let description = [{}];
  let cppNamespace = "::mlir::Plugin";
}

// Base class for Plugin dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Plugin_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Plugin_Dialect, mnemonic, traits>;

#endif // PLUGIN_DIALECT_TD