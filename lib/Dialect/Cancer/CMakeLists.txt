add_mlir_dialect_library(MLIRCancer
        CancerDialect.cpp
        CancerOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Cancer

        DEPENDS
        MLIRCancerOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
