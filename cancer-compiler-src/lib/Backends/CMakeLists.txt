# TODO IREE compilation may change into real compile control, currently, just symlinks
if(BUILD_CANCER_IREE_BACKEND)
  add_subdirectory(IREE)
endif()


# TODO add refjit
