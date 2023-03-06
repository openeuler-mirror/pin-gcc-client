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
        ArrayRef<Type> paramsType = Ty.getBody();
        for (auto ty :paramsType) {
            string paramStr = "elemType" + std::to_string(paramIndex++);
            item["structelemType"][paramStr] = TypeJsonSerialize(ty.dyn_cast<PluginIR::PluginTypeBase>());
        }
        paramIndex = 0;
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
        llvm::SmallVector<Type> typelist;
        Json::Value::Members elemTypeNum = type["structelemType"].getMemberNames();
        for (size_t paramIndex = 0; paramIndex < elemTypeNum.size(); paramIndex++) {
            string Key = "elemType" + std::to_string(paramIndex);
            mlir::Type paramTy = TypeJsonDeSerialize(type["structelemType"][Key].toStyledString(), context);
            typelist.push_back(paramTy);
        }
        llvm::SmallVector<StringRef> names;
        Json::Value::Members elemNameNum = type["structelemName"].getMemberNames();
        for (size_t paramIndex = 0; paramIndex < elemTypeNum.size(); paramIndex++) {
            string Key = "elemName" + std::to_string(paramIndex);
            StringRef elemName = type["structelemName"][Key].toStyledString();
            names.push_back(elemName);
        }
        baseType = PluginIR::PluginStructType::get(&context, tyName, typelist, names);
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

    root["id"] = std::to_string(cgnode.idAttr().getInt());
    root["attributes"]["order"] = std::to_string(cgnode.orderAttr().getInt());
    if (cgnode.definitionAttr().getValue()) {
        root["attributes"]["definition"] = "1";
    }else {
        root["attributes"]["definition"] = "0";
    }
    root["attributes"]["symbolName"] = cgnode.symbolNameAttr().getValue().str().c_str();
    fprintf(stderr, "dgy client cgnode json %s\n", root.toStyledString().c_str());
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
        item["id"] = std::to_string(d.idAttr().getInt());
        if (d.declaredInlineAttr().getValue()) {
            item["attributes"]["declaredInline"] = "1";
        } else {
            item["attributes"]["declaredInline"] = "0";
        }
        item["attributes"]["funcName"] = d.funcNameAttr().getValue().str().c_str();

        mlir::Type fnty = d.type();
        if (auto ty = fnty.dyn_cast<PluginIR::PluginFunctionType>()) {
            if (auto retTy = ty.dyn_cast<PluginIR::PluginTypeBase>()) {
                item["retType"] = TypeJsonSerialize(retTy);
            }
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
        bbId = op.addressAttr().getInt();
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["opCode"] = data.opCodeAttr().getValue().str().c_str();
    return root;
}

Json::Value PluginJson::RetOpJsonSerialize(RetOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    return root;
}

Json::Value PluginJson::DebugOpJsonSerialize(DebugOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    return root;
}

Json::Value PluginJson::FallThroughOpJsonSerialize(FallThroughOp data, uint64_t &bbId)
{
    Json::Value root;
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["destaddr"] = std::to_string(data.destaddrAttr().getInt());
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
        item["id"] = std::to_string(decl.idAttr().getInt());
        item["attributes"]["symName"] = decl.symNameAttr().getValue().str().c_str();
        item["attributes"]["typeID"] = decl.typeIDAttr().getInt();
        item["attributes"]["typeWidth"] = decl.typeWidthAttr().getInt();
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
        item["id"] = std::to_string(loop.idAttr().getInt());
        item["index"] = std::to_string(loop.indexAttr().getInt());
        item["attributes"]["innerLoopId"] = std::to_string(loop.innerLoopIdAttr().getInt());
        item["attributes"]["outerLoopId"] = std::to_string(loop.outerLoopIdAttr().getInt());
        item["attributes"]["numBlock"] = std::to_string(loop.numBlockAttr().getInt());
        operation = "loopOp" + std::to_string(i++);
        root[operation] = item;
        item.clear();
    }
    out = root.toStyledString();
}

void PluginJson::LoopOpJsonSerialize(mlir::Plugin::LoopOp& loop, string& out)
{
    Json::Value root;
    root["id"] = std::to_string(loop.idAttr().getInt());
    root["index"] = std::to_string(loop.indexAttr().getInt());
    root["attributes"]["innerLoopId"] = std::to_string(loop.innerLoopIdAttr().getInt());
    root["attributes"]["outerLoopId"] = std::to_string(loop.outerLoopIdAttr().getInt());
    root["attributes"]["numBlock"] = std::to_string(loop.numBlockAttr().getInt());
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
    item["id"] = std::to_string(data.idAttr().getInt());
    Optional<StringRef> calleeName = data.callee();
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
    item["id"] = std::to_string(data.idAttr().getInt());
    item["condCode"] = std::to_string(data.condCodeAttr().getInt());
    item["lhs"] = ValueJsonSerialize(data.GetLHS());
    item["rhs"] = ValueJsonSerialize(data.GetRHS());
    bbId = data.addressAttr().getInt();
    item["address"] = std::to_string(bbId);
    item["tbaddr"] = std::to_string(data.tbaddrAttr().getInt());
    item["fbaddr"] = std::to_string(data.fbaddrAttr().getInt());
    return item;
}

Json::Value PluginJson::PhiOpJsonSerialize(PhiOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["capacity"] = std::to_string(data.capacityAttr().getInt());
    item["nArgs"] = std::to_string(data.nArgsAttr().getInt());
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
        string input = "input" + std::to_string(opIdx++);
        item["operands"][input] = ValueJsonSerialize(v);
    }

    return item;
}

Json::Value PluginJson::SSAOpJsonSerialize(SSAOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["defCode"] = std::to_string(data.defCodeAttr().getInt());
    item["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    item["nameVarId"] = std::to_string(data.nameVarIdAttr().getInt());
    item["ssaParmDecl"] = std::to_string(data.ssaParmDeclAttr().getInt());
    item["version"] = std::to_string(data.versionAttr().getInt());
    item["definingId"] = std::to_string(data.definingIdAttr().getInt());
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginJson::AssignOpJsonSerialize(AssignOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["exprCode"] = std::to_string(data.exprCodeAttr().getInt());
    item["OperationName"] = data.getOperation()->getName().getStringRef().str();
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
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
        root["id"] = std::to_string(cOp.idAttr().getInt());
        root["defCode"] = std::to_string(static_cast<int32_t>(IDefineCode::IntCST));
        root["value"] = std::to_string(cOp.initAttr().cast<mlir::IntegerAttr>().getInt());
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
        root["id"] = std::to_string(phOp.idAttr().getInt());
        root["defCode"] = std::to_string(phOp.defCodeAttr().getInt());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["statement"] = data.statementAttr().getValue().str().c_str();
    root["nInputs"] = std::to_string(data.nInputsAttr().getInt());
    root["nOutputs"] = std::to_string(data.nOutputsAttr().getInt());
    root["nClobbers"] = std::to_string(data.nClobbersAttr().getInt());
    size_t opIdx = 0;
    for(mlir::Value v : data.operands()) {
        string index = std::to_string(opIdx++);
        root["operands"][index] = ValueJsonSerialize(v);
    }
    return root;
}

Json::Value PluginJson::SwitchOpJsonSerialize(mlir::Plugin::SwitchOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    bbId = data.addressAttr().getInt();
    size_t opIdx = 0;
    for(mlir::Value v : data.operands()) {
        string index = std::to_string(opIdx++);
        root["operands"][index] = ValueJsonSerialize(v);
    }
    root["defaultaddr"] = std::to_string(data.defaultaddrAttr().getInt());
    root["address"] = std::to_string(data.addressAttr().getInt());
    int index = 0;
    for (auto attr : data.caseaddrsAttr()) {
        root["case"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::ResxOpJsonSerialize(ResxOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["region"] = std::to_string(data.regionAttr().getInt());
    return root;
}

Json::Value PluginJson::BindOpJsonSerialize(BindOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["vars"] = ValueJsonSerialize(data.GetVars());
    root["block"] = ValueJsonSerialize(data.GetBlock());
    int index = 0;
    for (auto attr : data.bodyAttr()) {
        root["body"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::TryOpJsonSerialize(TryOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    int index = 0;
    for (auto attr : data.evalAttr()) {
        root["eval"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    index = 0;
    for (auto attr : data.cleanupAttr()) {
        root["cleanup"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    root["kind"] = std::to_string(data.kindAttr().getInt());
    return root;
}

Json::Value PluginJson::CatchOpJsonSerialize(CatchOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["types"] = ValueJsonSerialize(data.GetTypes());
    int index = 0;
    for (auto attr : data.handlerAttr()) {
        root["handler"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::EHDispatchOpJsonSerialize(EHDispatchOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["region"] = std::to_string(data.regionAttr().getInt());
    int index = 0;
    for (auto attr : data.ehHandlersaddrsAttr()) {
        root["ehHandlersaddrs"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return root;
}

Json::Value PluginJson::GotoOpJsonSerialize(GotoOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    root["dest"] = ValueJsonSerialize(data.GetLabel());
    root["successaddr"] = std::to_string(data.successaddrAttr().getInt());
    return root;
}

Json::Value PluginJson::TransactionOpJsonSerialize(TransactionOp data, uint64_t &bbId)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    bbId = data.addressAttr().getInt();
    root["address"] = std::to_string(bbId);
    int index = 0;
    for (auto attr : data.stmtaddrAttr()) {
        root["stmt"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    root["labelNorm"] = ValueJsonSerialize(data.GetTransactionNormal());
    root["labelUninst"] = ValueJsonSerialize(data.GetTransactionUinst());
    root["labelOver"] = ValueJsonSerialize(data.GetTransactionOver());

    root["fallthroughaddr"] = std::to_string(data.fallthroughaddrAttr().getInt());
    root["abortaddr"] = std::to_string(data.abortaddrAttr().getInt());
    return root;
}

Json::Value PluginJson::LabelOpJsonSerialize(LabelOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["label"] = ValueJsonSerialize(data.GetLabelLabel());
    return root;
}

Json::Value PluginJson::EHMntOpJsonSerialize(EHMntOp data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["decl"] = ValueJsonSerialize(data.Getfndecl());
    return root;
}

Json::Value PluginJson::NopOpJsonSerialize(NopOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    return item;
}

Json::Value PluginJson::EHElseOpJsonSerialize(EHElseOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());

    int index = 0;
    for (auto attr : data.nBodyAttr()) {
        item["nbody"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    index = 0;
    for (auto attr : data.eBodyAttr()) {
        item["ebody"][index] = std::to_string(attr.dyn_cast<IntegerAttr>().getInt());
    }
    return item;
}

Json::Value PluginJson::ConstructorOpJsonSerialize(ConstructorOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    root["len"] = std::to_string(data.lenAttr().getInt());
    int32_t i = 0;
    for (auto idx : data.idx()) {
        root["idx"][i++] = ValueJsonSerialize(idx);
    }
    i = 0;
    for (auto val : data.val()) {
        root["val"][i] = ValueJsonSerialize(val);
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::ListOpJsonSerialize(ListOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    root["hasPurpose"] = std::to_string(data.hasPurposeAttr().getValue());
    size_t opIdx = 0;
    for (mlir::Value v : data.operands()) {
        root["operands"][std::to_string(opIdx++)] = ValueJsonSerialize(v);
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::StrOpJsonSerialize(StrOp& data)
{
    Json::Value item;
    item["id"] = std::to_string(data.idAttr().getInt());
    item["defCode"] = std::to_string(data.defCodeAttr().getInt());
    item["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    item["str"] = data.strAttr().getValue().str();
    auto retTy = data.getType().dyn_cast<PluginIR::PluginTypeBase>();
    item["retType"] = TypeJsonSerialize(retTy);
    return item;
}

Json::Value PluginJson::ArrayOpJsonSerialize(ArrayOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    root["addressable"] = std::to_string(data.addressableAttr().getValue());
    root["used"] = std::to_string(data.usedAttr().getValue());
    root["uid"] = std::to_string(data.uidAttr().getInt());
    mlir::Value initial = data.GetInitial();
    mlir::Value name = data.GetName();
    if (data.GetChain()) {
        root["chain"] = std::to_string(data.chain().getValue());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    if (data.GetVars()) {
        root["vars"] = ValueJsonSerialize(data.GetVars());
    }
    if (data.GetSupercontext()) {
        root["supercontext"] = std::to_string(data.GetSupercontext().getValue());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    root["len"] = std::to_string(data.lenAttr().getInt());
    int index = 0;
    for (auto ele : data.elements()) {
        root["elements"][index] = ValueJsonSerialize(ele);
    }
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}

Json::Value PluginJson::FieldDeclOpJsonSerialize(FieldDeclOp& data)
{
    Json::Value root;
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    mlir::Value initial = data.GetInitial();
    mlir::Value name = data.GetName();
    mlir::Value fieldOffset = data.GetFieldOffset();
    mlir::Value fieldBitOffset = data.GetFieldBitOffset();
    root["initial"] = ValueJsonSerialize(initial);
    root["name"] = ValueJsonSerialize(name);
    if (data.GetChain()) {
        root["chain"] = std::to_string(data.chain().getValue());
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
    root["id"] = std::to_string(data.idAttr().getInt());
    root["defCode"] = std::to_string(data.defCodeAttr().getInt());
    root["readOnly"] = std::to_string(data.readOnlyAttr().getValue());
    mlir::Value operand = data.GetOperand();
    root["operand"] = ValueJsonSerialize(operand);
    auto retTy = data.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    root["retType"] = TypeJsonSerialize(retTy);
    return root;
}
} // namespace PinClient
