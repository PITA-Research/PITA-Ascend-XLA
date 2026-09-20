add_mlir_library(ConvertGraphdef
  export_graphdef.cc
  status.cc
)

target_link_libraries(ConvertGraphdef PUBLIC
  AirDialect
)