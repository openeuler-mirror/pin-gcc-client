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

   Author: Guangya Ding
   Create: 2022-11-23
   Description:
    This file declares the type translation function going from MLIR Plugin
    dialect to Plugin IR and back.
*/

#ifndef MLIR_TAGET_PLUGINIR_TYPETRANSLATION_H
#define MLIR_TAGET_PLUGINIR_TYPETRANSLATION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "Dialect/PluginTypes.h"

namespace PluginIR {
using std::vector;
using namespace mlir;

namespace detail {
    class TypeFromPluginIRTranslatorImpl;
    class TypeToPluginIRTranslatorImpl;
} // namespace detail

class TypeFromPluginIRTranslator {
public:
    TypeFromPluginIRTranslator (mlir::MLIRContext &context);
    ~TypeFromPluginIRTranslator ();

    /* Translates the given Plugin IR type to the MLIR Plugin dialect. */
    PluginTypeBase translateType (uintptr_t id);

    PluginTypeID getPluginTypeId (PluginTypeBase type);
  
    uint64_t getBitWidth (PluginTypeBase type);

private:
    std::unique_ptr<detail::TypeFromPluginIRTranslatorImpl> impl;
};

} // namespace PluginIR

#endif // MLIR_TAGET_PLUGINIR_TYPETRANSLATION_H