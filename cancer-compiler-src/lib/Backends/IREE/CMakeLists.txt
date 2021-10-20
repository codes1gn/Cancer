add_cancer_library(CANCERIREEBackend
    LowerLinkage.cpp
    Passes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SRC_DIR}/include/Backends/IREE

    DEPENDS
    CANCERIREEBackendPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    )

mlir_check_all_link_libraries(CANCERIREEBackend)
