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
    This file contains the declaration of the RegisterPluginEvent.
*/

#ifndef IRTRANS_PLUGIN_H
#define IRTRANS_PLUGIN_H

#include "PluginClient/PluginClient.h"
#include "tree-pass.h"
#include "tree.h"

int RegisterPluginEvent(PinClient::InjectPoint inject, const std::string& pluginName);
int RegisterPassManagerSetup(PinClient::InjectPoint inject, const PinClient::ManagerSetupData& setupData,
    const std::string& pluginName);
#endif
