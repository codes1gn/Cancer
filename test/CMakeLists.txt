configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CANCER_TEST_DEPENDS
        FileCheck count not
        cancer-opt
        cancer-translate
        cancer-runner
        cancer-runner-backend
        cancer_pyrunner
        )

add_lit_testsuite(check-cancer "Running the cancer regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CANCER_TEST_DEPENDS}
        )
set_target_properties(check-cancer PROPERTIES FOLDER "Tests")

add_lit_testsuites(CANCER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CANCER_TEST_DEPENDS})
