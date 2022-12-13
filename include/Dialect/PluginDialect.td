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

//===----------------------------------------------------------------------===//
// PluginDialect enum definitions
//===----------------------------------------------------------------------===//

def IComparisonLT    : I32EnumAttrCase<"lt", 0>;
def IComparisonLE    : I32EnumAttrCase<"le", 1>;
def IComparisonGT    : I32EnumAttrCase<"gt", 2>;
def IComparisonGE    : I32EnumAttrCase<"ge", 3>;
def IComparisonLTGT  : I32EnumAttrCase<"ltgt", 4>;
def IComparisonEQ    : I32EnumAttrCase<"eq", 5>;
def IComparisonNE    : I32EnumAttrCase<"ne", 6>;
def IComparisonUNDEF : I32EnumAttrCase<"UNDEF", 7>;
def IComparisonAttr  : I32EnumAttr<
   "IComparisonCode", "plugin comparison code",
   [IComparisonLT, IComparisonLE, IComparisonGT, IComparisonGE, 
    IComparisonLTGT, IComparisonEQ, IComparisonNE, IComparisonUNDEF]>{
  let cppNamespace = "::mlir::Plugin";
}

def IDefineCodeMemRef    : I32EnumAttrCase<"MemRef", 0>;
def IDefineCodeIntCST    : I32EnumAttrCase<"IntCST", 1>;
def IDefineCodeSSA       : I32EnumAttrCase<"SSA", 2>;
def IDefineCodeUNDEF     : I32EnumAttrCase<"UNDEF", 3>;
def IDefineCodeAttr  : I32EnumAttr<
   "IDefineCode", "plugin define code",
   [IDefineCodeMemRef, IDefineCodeIntCST, IDefineCodeUNDEF]>{
  let cppNamespace = "::mlir::Plugin";
}

def IExprCodePlus    : I32EnumAttrCase<"Plus", 0>;
def IExprCodeMinus   : I32EnumAttrCase<"Minus", 1>;
def IExprCodeMult    : I32EnumAttrCase<"Mult", 2>;
def IExprCodePtrPlus : I32EnumAttrCase<"PtrPlus", 3>;
def IExprCodeMin     : I32EnumAttrCase<"Min", 4>;
def IExprCodeMax     : I32EnumAttrCase<"Max", 5>;
def IExprCodeBitIOR  : I32EnumAttrCase<"BitIOR", 6>;
def IExprCodeBitXOR  : I32EnumAttrCase<"BitXOR", 7>;
def IExprCodeBitAND  : I32EnumAttrCase<"BitAND", 8>;
def IExprCodeLshift  : I32EnumAttrCase<"Lshift", 9>;
def IExprCodeRshift  : I32EnumAttrCase<"Rshift", 10>;
def IExprCodeNop     : I32EnumAttrCase<"Nop", 11>;
def IExprCodeUNDEF   : I32EnumAttrCase<"UNDEF", 12>;
def IExprCodeAttr  : I32EnumAttr<
   "IExprCode", "plugin expr code",
   [IExprCodePlus, IExprCodeMinus, IExprCodeMult,
    IExprCodePtrPlus, IExprCodeMin, IExprCodeMax,
    IExprCodeBitIOR, IExprCodeBitXOR, IExprCodeBitAND,
    IExprCodeLshift, IExprCodeRshift, IExprCodeNop,
    IExprCodeUNDEF]>{
  let cppNamespace = "::mlir::Plugin";
}

#endif // PLUGIN_DIALECT_TD