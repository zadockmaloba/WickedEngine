project(metal)
# Library definition
add_library(METAL_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/definition.cpp
        )

# Metal cpp headers
# target_include_directories(METAL_CPP SYSTEM PUBLIC
#         "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
#         "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-extensions"
#         )

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
        DESTINATION include/metal-cpp)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-extensions"
        DESTINATION include/metal-cpp-extensions)

target_include_directories(METAL_CPP PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-extensions>"
        "$<INSTALL_INTERFACE:include/metal-cpp>"
        "$<INSTALL_INTERFACE:include/metal-cpp-extensions>"
        )

# Metal cpp library (linker)
target_link_libraries(METAL_CPP
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework QuartzCore"
)
