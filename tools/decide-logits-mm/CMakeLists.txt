set(TARGET llama-decide-logits-mm)
add_executable(${TARGET} decide-logits-mm.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common mtmd llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
