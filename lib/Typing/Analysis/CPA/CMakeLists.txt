add_cancer_library(CANCERTypingCPA
  Algorithm.cpp
  Interfaces.cpp
  Types.cpp

  DEPENDS
  CANCERTypingCPAInterfacesIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
)
