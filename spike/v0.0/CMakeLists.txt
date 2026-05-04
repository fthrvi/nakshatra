cmake_minimum_required(VERSION 3.14)
project(nakshatra_v00_spike CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to llama.cpp checkout; override with -DLLAMA_DIR=...
set(LLAMA_DIR "/home/prithvi/llama.cpp" CACHE PATH "llama.cpp checkout")

function(add_spike_target name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${LLAMA_DIR}/include
        ${LLAMA_DIR}/ggml/include
    )
    target_link_directories(${name} PRIVATE ${LLAMA_DIR}/build/bin)
    target_link_libraries(${name} PRIVATE llama ggml ggml-cpu ggml-base)
    set_target_properties(${name} PROPERTIES
        INSTALL_RPATH "${LLAMA_DIR}/build/bin"
        BUILD_RPATH   "${LLAMA_DIR}/build/bin"
    )
endfunction()

add_spike_target(cb_eval_observe cb_eval_observe.cpp)
add_spike_target(cb_eval_write   cb_eval_write.cpp)
