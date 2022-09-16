cmake_minimum_required(VERSION 3.5.1)

project(Plugin C CXX)

include(cmake/common.cmake)
execute_process(COMMAND gcc -print-file-name=plugin 
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                TIMEOUT 3
                RESULT_VARIABLE result_var
                OUTPUT_VARIABLE output_var
                ERROR_VARIABLE error_var
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
include_directories(${output_var}/include)
include_directories(include)
add_compile_options(-std=c++11 -Wall -fPIC)

# Proto file
get_filename_component(plg_proto "protos/plugin.proto" ABSOLUTE)
get_filename_component(plg_proto_path "${plg_proto}" PATH)

# Generated sources
set(plg_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/plugin.pb.cc")
set(plg_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/plugin.pb.h")
set(plg_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/plugin.grpc.pb.cc")
set(plg_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/plugin.grpc.pb.h")
add_custom_command(
      OUTPUT "${plg_proto_srcs}" "${plg_proto_hdrs}" "${plg_grpc_srcs}" "${plg_grpc_hdrs}"
      COMMAND ${CMD_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${plg_proto_path}"
        --plugin=protoc-gen-grpc="${CMD_GRPC_CPP_PLUGIN}"
        "${plg_proto}"
      DEPENDS "${plg_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# plg_grpc_proto
add_library(plg_grpc_proto STATIC
  ${plg_grpc_srcs}
  ${plg_grpc_hdrs}
  ${plg_proto_srcs}
  ${plg_proto_hdrs})
target_link_libraries(plg_grpc_proto
  ${GRPC_PP_REFLECTION_LIBRARY}
  ${GRPC_PP_LIBRARY}
  ${PROTOBUF_LIBRARY})

add_library(pin_gcc_client SHARED "lib/PluginClient/PluginLog.cpp" "lib/PluginClient/PluginClient.cpp" "lib/IRTrans/IRTransPlugin.cpp" "lib/Conversion/GimpleToPlugin.cpp" "lib/PluginAPI/PluginAPI.cpp")
target_link_libraries(pin_gcc_client
  ${JSONCPP_LIBRARY}
  plg_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# install
install(TARGETS pin_gcc_client LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/configs/pin-gcc-client.json" DESTINATION bin)

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
