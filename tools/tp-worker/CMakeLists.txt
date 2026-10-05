if (NOT BITNET_ARM_TL1)
    message(WARNING "BITNET_TP is ON but BITNET_ARM_TL1 is OFF -- bitnet-tp-worker "
                     "needs the TL1 kernel (ggml_preprocessor/ggml_qgemm_lut) and will "
                     "fail to compile without -DBITNET_ARM_TL1=ON.")
endif()

add_executable(bitnet-tp-worker main.cpp)

target_include_directories(bitnet-tp-worker PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tools/tp-common
    ${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/vendor/nlohmann)

target_link_libraries(bitnet-tp-worker PRIVATE ggml Threads::Threads)
target_compile_features(bitnet-tp-worker PRIVATE cxx_std_17)
