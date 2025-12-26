file(GLOB_RECURSE EAGER_GRAPH_BUILDER CONFIGURE_DEPENDS "*.cpp")

add_library(eager_graph_builder STATIC ${EAGER_GRAPH_BUILDER})

target_compile_options(eager_graph_builder
    PRIVATE
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        #        -Werror
        -Wno-subobject-linkage
        )

target_link_libraries(eager_graph_builder PRIVATE
        intf_pub
        slog_headers
        mmpa_headers
        metadef_headers
        c_sec
        -ldl
)

add_library(eager_graph_builder_lib INTERFACE)
target_link_libraries(eager_graph_builder_lib INTERFACE eager_graph_builder)
target_include_directories(eager_graph_builder_lib INTERFACE ${CMAKE_CURRENT_LIST_DIR})
