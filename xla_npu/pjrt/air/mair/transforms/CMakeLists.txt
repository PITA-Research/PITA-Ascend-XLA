# The outer CMake needs to rely on MLIRMairPassIncGen to generate mair_passes.h.inc
set(LLVM_TARGET_DEFINITIONS mair_passes.td)
mlir_tablegen(mair_passes.h.inc -gen-pass-decls -name Mair)
add_public_tablegen_target(MLIRMairPassIncGen)

add_mlir_library(MairPasses
        mair_passes.cc

        DEPENDS
        MLIRMairPassIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        AirDialect

        MLIRIR
        MLIRPass
)
