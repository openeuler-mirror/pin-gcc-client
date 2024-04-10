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
    This file contains the implementation of the PluginJson class.
*/

#include <iostream>
#include <json/json.h>
#include "PluginAPI/PluginClientAPI.h"
#include "PluginClient/PluginLog.h"
#include "PluginClient/PluginJson.h"

namespace PinClient {
using std::map;
using namespace mlir::Plugin;
using namespace mlir;

static uintptr_t GetID(Json::Value node)
{
    string id = node.asString();
    return atol(id.c_str());
}

Json::Value PluginJson::TypeJsonSerialize(PluginIR::PluginTypeBase type)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    uint64_t ReTypeId;
    uint64_t ReTypeWidth;

    ReTypeId = static_cast<uint64_t>(type.getPluginTypeID());
    item["id"] = std::to_string(ReTypeId);

    if (auto Ty = type.dyn_cast<PluginIR::PluginStructType>()) {
        std::string tyName = Ty.getName().str();
        item["structtype"] = tyName;
        size_t paramIndex = 0;
        ArrayRef<StringRef> paramsNames = Ty.getElementNames();
        for (auto name :paramsNames) {
            string paramStr = "elemName" + std::to_string(paramIndex++);
            item["structelemName"][paramStr] = name.str();
        }
    }

    if (auto Ty = type.dyn_cast<PluginIR::PluginFunctionType>()) {
        auto fnrestype = Ty.getReturnType().dyn_cast<PluginIR::PluginTypeBase>();
        item["fnreturntype"] = TypeJsonSerialize(fnrestype);
        size_t paramIndex = 0;
        ArrayRef<Type> paramsType = Ty.getParams();
        for (auto ty : Ty.getParams()) {
            string paramStr = "argType" + std::to_string(paramIndex++);
            item["fnargsType"][paramStr] = TypeJsonSerialize(ty.dyn_cast<PluginIR::PluginTypeBase>());
        }
    }

    if (auto Ty = type.dyn_cast<PluginIR::PluginVectorType>()) {
        auto elemTy = Ty.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        item["elementType"] = TypeJsonSerialize(elemTy);
        uint64_t elemNum = Ty.getNumElements();
        item["vectorelemnum"] = std::to_string(elemNum);
    }

    if (auto Ty = type.dyn_cast<PluginIR::PluginArrayType>()) {
        auto elemTy = Ty.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        item["elementType"] = TypeJsonSerialize(elemTy);
        uint64_t elemNum = Ty.getNumElements();
        item["arraysize"] = std::to_string(elemNum);
    }

    if (auto elemTy = type.dyn_cast<PluginIR::PluginPointerType>()) {
        auto baseTy = elemTy.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        item["elementType"] = TypeJsonSerialize(baseTy);
        if (elemTy.isReadOnlyElem()) {
            item["elemConst"] = "1";
        } else {
            item["elemConst"] = "0";
        }
    }

    if (type.getPluginIntOrFloatBitWidth() != 0) {
        ReTypeWidth = type.getPluginIntOrFloatBitWidth();
        item["width"] = std::to_string(ReTypeWidth);
    }

    if (type.isSignedPluginInteger()) {
        item["signed"] = "1";
    }

    if (type.isUnsignedPluginInteger()) {
        item["signed"] = "0";
    }

    root["type"] = item;
    return root;
}

PluginIR::PluginTypeBase PluginJson::TypeJsonDeSerialize(const string& data, mlir::MLIRContext &context)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    PluginIR::PluginTypeBase baseType;

    Json::Value type = root["type"];
    uint64_t id = GetID(type["id"]);
    PluginIR::PluginTypeID PluginTypeId = static_cast<PluginIR::PluginTypeID>(id);

    if (type["signed"] && (id >= static_cast<uint64_t>(PluginIR::UIntegerTy1ID)
        && id <= static_cast<uint64_t>(PluginIR::IntegerTy64ID))) {
        string s = type["signed"].asString();
        uint64_t width = GetID(type["width"]);
        if (s == "1") {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Signed);
        } else {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Unsigned);
        }
    } else if (type["width"] && (id == static_cast<uint64_t>(PluginIR::FloatTyID)
        || id == static_cast<uint64_t>(PluginIR::DoubleTyID))) {
        uint64_t width = GetID(type["width"]);
        baseType = PluginIR::PluginFloatType::get(&context, width);
    } else if (id == static_cast<uint64_t>(PluginIR::PointerTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString(), context);
        baseType = PluginIR::PluginPointerType::get(&context, elemTy, type["elemConst"].asString() == "1" ? 1 : 0);
    } else if (id == static_cast<uint64_t>(PluginIR::ArrayTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString(), context);
        uint64_t elemNum = GetID(type["arraysize"]);
        baseType = PluginIR::PluginArrayType::get(&context, elemTy, elemNum);
    } else if (id == static_cast<uint64_t>(PluginIR::FunctionTyID)) {
        mlir::Type returnTy = TypeJsonDeSerialize(type["fnreturntype"].toStyledString(), context);
        llvm::SmallVector<Type> typelist;
        Json::Value::Members fnTypeNum = type["fnargsType"].getMemberNames();
        uint64_t argsNum = fnTypeNum.size();
        for (size_t paramIndex = 0; paramIndex < argsNum; paramIndex++) {
            string Key = "argType" + std::to_string(paramIndex);
            mlir::Type paramTy = TypeJsonDeSerialize(type["fnargsType"][Key].toStyledString(), context);
            typelist.push_back(paramTy);
        }
        baseType = PluginIR::PluginFunctionType::get(&context, returnTy, typelist);
    } else if (id == static_cast<uint64_t>(PluginIR::StructTyID)) {
        StringRef tyName = type["structtype"].toStyledString();
        llvm::SmallVector<StringRef> names;
        Json::Value::Members elemNameNum = type["structelemName"].getMemberNames();
        for (size_t paramIndex = 0; paramIndex < elemNameNum.size(); paramIndex++) {
            string Key = "elemName" + std::to_string(paramIndex);
            StringRef elemName = type["structelemName"][Key].toStyledString();
            names.push_back(elemName);
        }
        baseType = PluginIR::PluginStructType::get(&context, tyName, names);
    }
    else {
        if (PluginTypeId == PluginIR::VoidTyID) {
            baseType = PluginIR::PluginVoidType::get(&context);
        }
        if (PluginTypeId == PluginIR::BooleanTyID) {
            baseType = PluginIR::PluginBooleanType::get(&context);
        }
        if (PluginTypeId == PluginIR::UndefTyID) {
            baseType = PluginIR::PluginUndefType::get(&context);
        }
    }

    return baseType;
}

void PluginJson::CGnodeOpJsonSerialize(CGnodeOp& cgnode, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    root["id"] = std::to_string(cgnode.getIdAttr().getInt());
    root["attributes"]["order"] = std::to_string(cgnode.getOrderAttr().getInt());
    if (cgnode.getDefinitionAttr().getValue()) {
        root["attributes"]["definition"] = "1";
    }else {
        root["attributes"]["definition"] = "0";
    }
    root["attributes"]["symbolName"] = cgnode.getSymbolNameAttr().getValue().str().c_str();
    out = root.toStyledString();
}

void PluginJson::FunctionOpJsonSerialize(vector<FunctionOp>& data, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    int i = 0;
    string operation;
    for (auto& d: data) {
        item["id"] = std::to_string(d.getIdAttr().getInt());
        if (d.getDeclaredInlineAttr().getValue()) {
            item["attributes"]["declaredInline"] = "1";
        } else {
            item["attributes"]["declaredInline"] = "0";
        }
        item["attributes"]["funcName"] = d.getFuncNameAttr().getValue().str().c_str();

        if (d.getValidTypeAttr().getValue()) {
            item["attributes"]["validType"] = "1";
            mlir::Type fnty = d.getType();
            if (auto ty = fnty.dyn_cast<PluginIR::PluginFunctionType>()) {
                if (auto retTy = ty.dyn_cast<PluginIR::PluginTypeBase>()) {
                    item["retType"] = TypeJsonSerialize(retTy);
                }
            }
        } else {
            item["attributes"]["validType"] = "0";
        }

        auto &region = d.getRegion();
        size_t bbIdx = 0;
        for (auto &b : region) {
            string blockStr = "block" + std::to_string(bbIdx++);
            uint64_t bbAddress = 0;
            size_t opIdx = 0;
            for (auto &inst : b) {
                if (isa<PlaceholderOp>(inst)) {
                    continue;
                } else if (isa<SSAOp>(inst)) {
                    continue;
                } else if (isa<MemOp>(inst)) {
                    continue;
                } else if (isa<ConstOp>(inst)) {
                    continue;
                } else if (isa<ListOp>(inst)) {
					continue;
				} else if (isa<StrOp>(inst)) {
					continue;
				} else if (isa<ArrayOp>(inst)) {
					continue;
				} else if (isa<DeclBaseOp>(inst)) {
					continue;
				} else if (isa<FieldDeclOp>(inst)) {
					continue;
				} else if (isa<ConstructorOp>(inst)) {
					continue;
				} else if (isa<VecOp>(inst)) {
					continue;
				} else if (isa<BlockOp>(inst)) {
					continue;
				} else if (isa<AddressOp>(inst)) {
					continue;
				}
				string opStr = "Operation" + std::to_string(opIdx++);
                item["region"][blockStr]["ops"][opStr] = OperationJsonSerialize(&inst, bbAddress);
            }
            assert(bbAddress != 0);
            item["region"][blockStr]["address"] = std::to_string(bbAddress);
        }
        operation = "FunctionOp" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

Json::Value PluginJson::OperationJsonSerialize(mlir::Operation *operation, uint64_t &bbId)
{
    Json::Value root;
    if (AssignOp op = llvm::dyn_cast<AssignOp>(operation)) {
        root = AssignOpJsonSerialize(op);
    } else if (CallOp op = llvm::dyn_cast<CallOp>(operation)) {
        root = CallOpJsonSerialize(op);
        bbId = op.getAddressAttr().getInt();
    } else if (CondOp op = llvm::dyn_cast<CondOp>(operation)) {
        root = CondOpJsonSerialize(op, bbId);
    } else if (PhiOp op = llvm::dyn_cast<PhiOp>(operation)) {
        root = PhiOpJsonSerialize(op);
    } else if (FallThroughOp op = llvm::dyn_cast<FallThroughOp>(operation)) {
        root = FallThroughOpJsonSerialize(op, bbId);
    } else if (RetOp op = llvm::dyn_cast<RetOp>(operation)) {
        root = RetOpJsonSerialize(op, bbId);
    } else if (BaseOp op = llvm::dyn_cast<BaseOp>(operation)) {
        root = BaseOpJsonSerialize(op);
    } else if (DebugOp op = llvm::dyn_cast<DebugOp>(operation)) {
        root = DebugOpJsonSerialize(op);
    } else if (AsmOp op = llvm::dyn_cast<AsmOp>(operation)) {
        root = AsmOpJsonSerialize(op);
    } else if (SwitchOp op = llvm::dyn_cast<SwitchOp>(operation)) {
        root = SwitchOpJsonSerialize(op, bbId);
    } else if (GotoOp op = llvm::dyn_cast<GotoOp>(operation)) {
        root = GotoOpJsonSerialize(op, bbId);
    } else if (LabelOp op = llvm::dyn_cast<LabelOp>(operation)) {
        root = LabelOpJsonSerialize(op);
    } else if (TransactionOp op = llvm::dyn_cast<TransactionOp>(operation)) {
        root = TransactionOpJsonSerialize(op, bbId);
    } else if (ResxOp op = llvm::dyn_cast<ResxOp>(operation)) {
        root = ResxOpJsonSerialize(op, bbId);
    } else if (EHMntOp op = llvm::dyn_cast<EHMntOp>(operation)) {
        root = EHMntOpJsonSerialize(op);
    } else if (EHDispatchOp op = llvm::dyn_cast<EHDispatchOp>(operation)) {
        root = EHDispatchOpJsonSerialize(op, bbId);
    } else if (BindOp op = llvm::dyn_cast<BindOp>(operation)) {
        root = BindOpJsonSerialize(op);
    } else if (TryOp op = llvm::dyn_cast<TryOp>(operation)) {
        root = TryOpJsonSerialize(op);
    } else if (CatchOp op = llvm::dyn_cast<CatchOp>(operation)) {
        root = CatchOpJsonSerialize(op);
    } else if (NopOp op = llvm::dyn_cast<NopOp>(operation)) {
        root = NopOpJsonSerialize(op);
    } else if (EHElseOp op = llvm::dyn_cast<EHElseOp>(operation)) {
        root = EHElseOpJsonSerialize(op);
    }
    root["OperationName"] = operation->getName().getStringRef().str();
    return root;
}

Json::Value PluginJson::BaseOpJsonSerialize(BaseOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["opCode"] = data.getOpCodeAttr().getValue().str().c_str();
    return root;
}

Json::Value PluginJson::RetOpJsonSerialize(RetOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.getAddressAttr().getInt();
    root["address"] = std::to_string(bbId);
    return root;
}

Json::Value PluginJson::DebugOpJsonSerialize(DebugOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    return root;
}

Json::Value PluginJson::FallThroughOpJsonSerialize(FallThroughOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.getAddressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["destaddr"] = std::to_string(data.getDestaddrAttr().getInt());
    return root;
}

void PluginJson::LocalDeclsJsonSerialize(vector<LocalDeclOp>& decls, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto& decl: decls) {
        item["id"] = std::to_string(decl.getIdAttr().getInt());
        item["attributes"]["symName"] = decl.getSymNameAttr().getValue().str().c_str();
        item["attributes"]["typeID"] = decl.getTypeIDAttr().getInt();
        item["attributes"]["typeWidth"] = decl.getTypeWidthAttr().getInt();
        operation = "localDecl" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::FunctionDeclsJsonSerialize(vector<mlir::Plugin::DeclBaseOp>& decls, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto& decl: decls) {
        item = DeclBaseOpJsonSerialize(decl);
        operation = std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::FiledOpsJsonSerialize(vector<mlir::Plugin::FieldDeclOp>& decls, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto& decl: decls) {
        item = FieldDeclOpJsonSerialize(decl);
        operation = std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::LoopOpsJsonSerialize(vector<mlir::Plugin::LoopOp>& loops, string& out)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;
    int i = 0;
    string operation;

    for (auto& loop: loops) {
        item["id"] = std::to_string(loop.getIdAttr().getInt());
        item["index"] = std::to_string(loop.getIndexAttr().getInt());
        item["attributes"]["innerLoopId"] = std::to_string(loop.getInnerLoopIdAttr().getInt());
        item["attributes"]["outerLoopId"] = std::to_string(loop.getOuterLoopIdAttr().getInt());
        item["attributes"]["numBlock"] = std::to_string(loop.getNumBlockAttr().getInt());
        operation = "loopOp" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out)
{
    Json::Value root;
    root["id"] = std::to_string(loop.getIdAttr().getInt());
    root["index"] = std::to_string(loop.getIndexAttr().getInt());
    root["attributes"]["innerLoopId"] = std::to_string(loop.getInnerLoopIdAttr().getInt());
    root["attributes"]["outerLoopId"] = std::to_string(loop.getOuterLoopIdAttr().getInt());
    root["attributes"]["numBlock"] = std::to_string(loop.getNumBlockAttr().getInt());
    out = root.toStyledString();
}

void PluginJson::OpsJsonSerialize(vector<std::pair<mlir::Operation*, uint64_t>>& ops, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for (auto& op : ops) {
        item = OperationJsonSerialize(op.first, op.second);
        index = "Op" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::ValuesJsonSerialize(vector<mlir::Value>& values, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for (auto& v : values) {
        item = ValueJsonSerialize(v);
        index = "Value" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::IDsJsonSerialize(vector<uint64_t>& ids, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for (auto& id : ids) {
        item["id"] = std::to_string(id);
        index = "ID" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::EdgesJsonSerialize(vector<std::pair<uint64_t, uint64_t> >& edges, string& out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string index;

    for (auto& edge : edges) {
        item["src"] = std::to_string(edge.first);
        item["dest"] = std::to_string(edge.second);
        index = "edge" + std::to_string(i++);
        root[index] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::EdgeJsonSerialize(std::pair<uint64_t, uint64_t>& edge, string& out)
{
    Json::Value root;
    root["src"] = std::to_string(edge.first);
    root["dest"] = std::to_string(edge.second);
    out = root.toStyledString();
}

// void类型的Json序列化
void PluginJson::NopJsonSerialize(string& out)
{
    Json::Value root;
    out = root.toStyledString();
}

void PluginJson::GetPhiOpsJsonSerialize(vector<PhiOp> phiOps, string & out)
{
    Json::Value root;
    Json::Value item;
    int i = 0;
    string operation;
    uint64_t placeholder = 0;
    for (auto phi : phiOps) {
        item = OperationJsonSerialize(phi.getOperation(), placeholder);
        operation = "operation" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

Json::Value PluginJson::CallOpJsonSerialize(CallOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    std::optional<StringRef> calleeName = data.getCallee();
    if (calleeName) {
        item["callee"] = calleeName->str();
    }    
	item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.getArgOperands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::CondOpJsonSerialize(CondOp& data, uint64_t &bbId)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    item["condCode"] = std::to_string(data.getCondCodeAttr().getInt());
    item["lhs"] = ValueJsonSerialize(data.GetLHS());
    item["rhs"] = ValueJsonSerialize(data.GetRHS());
    bbId = data.getAddressAttr().getInt();
    item["address"] = std::to_string(bbId);
    item["tbaddr"] = std::to_string(data.getTbaddrAttr().getInt());
    item["fbaddr"] = std::to_string(data.getFbaddrAttr().getInt());
    return item;
}

Json::Value PluginJson::PhiOpJsonSerialize(PhiOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    item["capacity"] = std::to_string(data.getCapacityAttr().getInt());
    item["nArgs"] = std::to_string(data.getNArgsAttr().getInt());
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.getOperands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::SSAOpJsonSerialize(SSAOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    item["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    item["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    item["nameVarId"] = std::to_string(data.getNameVarIdAttr().getInt());
    item["ssaParmDecl"] = std::to_string(data.getSsaParmDeclAttr().getInt());
    item["version"] = std::to_string(data.getVersionAttr().getInt());
    item["definingId"] = std::to_string(data.getDefiningIdAttr().getInt());
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginJson::AssignOpJsonSerialize(AssignOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    item["exprCode"] = std::to_string(data.getExprCodeAttr().getInt());
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.getOperands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::ValueJsonSerialize(mlir::Value data)
{
    Json::Value root;
    if (ConstOp cOp = data.getDefiningOp<ConstOp>()) {
        auto retTy = data.getType().dyn_cast<PluginIR::PluginTypeBase>();
        root["retType"] = TypeJsonSerialize(retTy);
        root["id"] = std::to_string(cOp.getIdAttr().getInt());
        root["defCode"] = std::to_string(static_cast<int32_t>(IDefineCode::IntCST));
        root["value"] = std::to_string(cOp.getInitAttr().cast<mlir::IntegerAttr>().getInt());
    } else if (MemOp mOp = data.getDefiningOp<MemOp>()) {
        root = MemOpJsonSerialize(mOp);
    } else if (SSAOp sOp = data.getDefiningOp<SSAOp>()) {
        root = SSAOpJsonSerialize(sOp);
    } else if (ListOp sOp = data.getDefiningOp<ListOp>()) {
        root = ListOpJsonSerialize(sOp);
    } else if (StrOp sOp = data.getDefiningOp<StrOp>()) {
        root = StrOpJsonSerialize(sOp);
    } else if (ArrayOp sOp = data.getDefiningOp<ArrayOp>()) {
        root = ArrayOpJsonSerialize(sOp);
    } else if (DeclBaseOp sOp = data.getDefiningOp<DeclBaseOp>()) {
        root = DeclBaseOpJsonSerialize(sOp);
    } else if (FieldDeclOp sOp = data.getDefiningOp<FieldDeclOp>()) {
        root = FieldDeclOpJsonSerialize(sOp);
    } else if (AddressOp sOp = data.getDefiningOp<AddressOp>()) {
        root = AddressOpJsonSerialize(sOp);
    } else if (ComponentOp sOp = data.getDefiningOp<ComponentOp>()) {
        root = ComponentOpJsonSerialize(sOp);
    } else if (ConstructorOp sOp = data.getDefiningOp<ConstructorOp>()) {
        root = ConstructorOpJsonSerialize(sOp);
    } else if (VecOp sOp = data.getDefiningOp<VecOp>()) {
        root = VecOpJsonSerialize(sOp);
    } else if (BlockOp sOp = data.getDefiningOp<BlockOp>()) {
        root = BlockOpJsonSerialize(sOp);
    } else if (PlaceholderOp phOp = data.getDefiningOp<PlaceholderOp>()) {
        root["id"] = std::to_string(phOp.getIdAttr().getInt());
        root["defCode"] = std::to_string(phOp.getDefCodeAttr().getInt());
        auto retTy = phOp.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
        root["retType"] = TypeJsonSerialize(retTy);
    } else {
        LOGE("ERROR: Can't Serialize!");
    }
    return root;
}

Json::Value PluginJson::MemOpJsonSerialize(MemOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    mlir::Value base = data.GetBase();
    mlir::Value offset = data.GetOffset();
    root["base"] = ValueJsonSerialize(base);
    root["offset"] = ValueJsonSerialize(offset);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

void PluginJson::IntegerSerialize(int64_t data, string& out)
{
    Json::Value root;
    root["integerData"] = data;
    out = root.toStyledString();
}

void PluginJson::StringSerialize(const string& data, string& out)
{
    Json::Value root;
    root["stringData"] = data;
    out = root.toStyledString();
}

Json::Value PluginJson::AsmOpJsonSerialize(mlir::Plugin::AsmOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["statement"] = data.getStatementAttr().getValue().str().c_str();
    root["nInputs"] = std::to_string(data.getNInputsAttr().getInt());
    root["nOutputs"] = std::to_string(data.getNOutputsAttr().getInt());
    root["nClobbers"] = std::to_string(data.getNClobbersAttr().getInt());
    size_t opIdx = 0;
    for(mlir::Value v : data.getOperands()) {
        string index = std::to_string(opIdx++);
        root["operands"][index] = ValueJsonSerialize(v);
    }
    return root;
}

Json::Value PluginJson::SwitchOpJsonSerialize(mlir::Plugin::SwitchOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    bbId = data.getAddressAttr().getInt();
    size_t opIdx = 0;
    for(mlir::Value v : data.getOperands()) {
        string index = std::to_string(opIdx++);
        root["operands"][index] = ValueJsonSerialize(v);
    }
    root["defaultaddr"] = std::to_string(data.getDefaultaddrAttr().getInt());
    root["address"] = std::to_string(data.getAddressAttr().getInt());
    int index = 0;
    for (auto attr : data.getCaseaddrsAttr()) {
        root["case"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::ResxOpJsonSerialize(ResxOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    bbId = data.getAddressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["region"] = std::to_string(data.getRegionAttr().getInt());
    return root;
}

Json::Value PluginJson::BindOpJsonSerialize(BindOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["vars"] = ValueJsonSerialize(data.GetVars());
    root["block"] = ValueJsonSerialize(data.GetBlock());
    int index = 0;
    for (auto attr : data.getBodyAttr()) {
        root["body"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::TryOpJsonSerialize(TryOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    int index = 0;
    for (auto attr : data.getEvalAttr()) {
        root["eval"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    index = 0;
    for (auto attr : data.getCleanupAttr()) {
        root["cleanup"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    root["kind"] = std::to_string(data.getKindAttr().getInt());
    return root;
}

Json::Value PluginJson::CatchOpJsonSerialize(CatchOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["types"] = ValueJsonSerialize(data.GetTypes());
    int index = 0;
    for (auto attr : data.getHandlerAttr()) {
        root["handler"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::EHDispatchOpJsonSerialize(EHDispatchOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    bbId = data.getAddressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["region"] = std::to_string(data.getRegionAttr().getInt());
    int index = 0;
    for (auto attr : data.getEhHandlersaddrsAttr()) {
        root["ehHandlersaddrs"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::GotoOpJsonSerialize(GotoOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    bbId = data.getAddressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["dest"] = ValueJsonSerialize(data.GetLabel());
    root["successaddr"] = std::to_string(data.getSuccessaddrAttr().getInt());
    return root;
}

Json::Value PluginJson::TransactionOpJsonSerialize(TransactionOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    bbId = data.getAddressAttr().getInt();
    root["address"] = std::to_string(bbId);
    int index = 0;
    for (auto attr : data.getStmtaddrAttr()) {
        root["stmt"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    root["labelNorm"] = ValueJsonSerialize(data.GetTransactionNormal());
    root["labelUninst"] = ValueJsonSerialize(data.GetTransactionUinst());
    root["labelOver"] = ValueJsonSerialize(data.GetTransactionOver());

    root["fallthroughaddr"] = std::to_string(data.getFallthroughaddrAttr().getInt());
    root["abortaddr"] = std::to_string(data.getAbortaddrAttr().getInt());
    return root;
}

Json::Value PluginJson::LabelOpJsonSerialize(LabelOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["label"] = ValueJsonSerialize(data.GetLabelLabel());
    return root;
}

Json::Value PluginJson::EHMntOpJsonSerialize(EHMntOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["decl"] = ValueJsonSerialize(data.Getfndecl());
    return root;
}

Json::Value PluginJson::NopOpJsonSerialize(NopOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    return item;
}

Json::Value PluginJson::EHElseOpJsonSerialize(EHElseOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());

    int index = 0;
    for (auto attr : data.getNBodyAttr()) {
        item["nbody"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    index = 0;
    for (auto attr : data.getEBodyAttr()) {
        item["ebody"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return item;
}

Json::Value PluginJson::ConstructorOpJsonSerialize(ConstructorOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    root["len"] = std::to_string(data.getLenAttr().getInt());
    int32_t i = 0;
    for (auto idx : data.getIdx()) {
        root["idx"][i++] = ValueJsonSerialize(idx);
    }
    i = 0;
    for (auto val : data.getVal()) {
        root["val"][i] = ValueJsonSerialize(val);
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::ListOpJsonSerialize(ListOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    root["hasPurpose"] = std::to_string(data.getHasPurposeAttr().getValue());
    size_t opIdx = 0;
    for (mlir::Value v : data.getOperands()) {
        root["operands"][std::to_string(opIdx++)] = ValueJsonSerialize(v);
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::StrOpJsonSerialize(StrOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.getIdAttr().getInt());
    item["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    item["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    item["str"] = data.getStrAttr().getValue().str();
    auto retTy = data.getType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginJson::ArrayOpJsonSerialize(ArrayOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    mlir::Value base = data.GetBase();
    mlir::Value offset = data.GetOffset();
    root["base"] = ValueJsonSerialize(base);
    root["offset"] = ValueJsonSerialize(offset);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::ComponentOpJsonSerialize(ComponentOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    mlir::Value component = data.GetComponent();
    mlir::Value field = data.GetField();
    root["component"] = ValueJsonSerialize(component);
    root["field"] = ValueJsonSerialize(field);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::DeclBaseOpJsonSerialize(DeclBaseOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    root["addressable"] = std::to_string(data.getAddressableAttr().getValue());
    root["used"] = std::to_string(data.getUsedAttr().getValue());
    root["uid"] = std::to_string(data.getUidAttr().getInt());
    mlir::Value initial = data.GetInitial();
    mlir::Value name = data.GetName();
    if (data.GetChain()) {
        root["chain"] = std::to_string(data.getChain().value());
    }
    root["initial"] = ValueJsonSerialize(initial);
    root["name"] = ValueJsonSerialize(name);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::BlockOpJsonSerialize(BlockOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    if (data.GetVars()) {
        root["vars"] = ValueJsonSerialize(data.GetVars());
    }
    if (data.GetSupercontext()) {
        root["supercontext"] = std::to_string(data.GetSupercontext().value());
    }
    if (data.GetSubblocks()) {
        root["subblocks"] = ValueJsonSerialize(data.GetSubblocks());
    }
    if (data.GetChain()) {
        root["chain"] = ValueJsonSerialize(data.GetChain());
    }
    if (data.GetAbstractorigin()) {
        root["abstract_origin"] = ValueJsonSerialize(data.GetAbstractorigin());
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::VecOpJsonSerialize(VecOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    root["len"] = std::to_string(data.getLenAttr().getInt());
    int index = 0;
    for (auto ele : data.getElements()) {
        root["elements"][index] = ValueJsonSerialize(ele);
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::FieldDeclOpJsonSerialize(FieldDeclOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    mlir::Value initial = data.GetInitial();
    mlir::Value name = data.GetName();
    mlir::Value fieldOffset = data.GetFieldOffset();
    mlir::Value fieldBitOffset = data.GetFieldBitOffset();
    root["initial"] = ValueJsonSerialize(initial);
    root["name"] = ValueJsonSerialize(name);
    if (data.GetChain()) {
        root["chain"] = std::to_string(data.getChain().value());
    }
    root["fieldOffset"] = ValueJsonSerialize(fieldOffset);
    root["fieldBitOffset"] = ValueJsonSerialize(fieldBitOffset);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::AddressOpJsonSerialize(AddressOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.getIdAttr().getInt());
    root["defCode"] = std::to_string(data.getDefCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.getReadOnlyAttr().getValue());
    mlir::Value operand = data.GetOperand();
    root["operand"] = ValueJsonSerialize(operand);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}
} // namespace PinClient
