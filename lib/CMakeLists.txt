add_subdirectory(Dialect)

add_mlir_library(MLIRClientAPI
  Translate/TypeTranslation.cpp
  Translate/GimpleToPluginOps.cpp
  PluginAPI/PluginClientAPI.cpp

  DEPENDS
  MLIRPluginOpsIncGen
  MLIRPlugin

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPlugin
)