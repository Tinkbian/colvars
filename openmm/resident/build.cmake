cmake_minimum_required(VERSION 3.24)
project(OpenMMColvars LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(OPENMM_COLVARS_ENABLE_CUDA "Build the native CUDA/Colvars device bridge" OFF)
option(OPENMM_COLVARS_BUILD_PYTHON "Build Python bindings" ON)
set(OPENMM_ROOT "" CACHE PATH "OpenMM prefix containing include and lib")
set(OPENMM_SOURCE_DIR "" CACHE PATH "Matching OpenMM sources (for optional Lepton)")
set(OPENMM_COLVARS_PYTHON_INSTALL_DIR "" CACHE PATH "Python installation directory")
get_filename_component(COLVARS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(COLVARS_CORE_DIR "${COLVARS_SOURCE_DIR}/src")

# Older versions cached automatic find_path/find_library results in these
# variables. Discard those results so changing Python or OPENMM_ROOT works
# on reconfigure. A fresh -D override has its own CMake cache help string.
foreach(var IN ITEMS OPENMM_INCLUDE_DIR OPENMM_LIBRARY OPENMM_CUDA_LIBRARY)
    get_property(old_help CACHE "${var}" PROPERTY HELPSTRING)
    if(old_help STREQUAL "Path to a file." OR old_help STREQUAL "Path to a library.")
        unset(${var} CACHE)
    endif()
endforeach()
set(OPENMM_INCLUDE_DIR "" CACHE PATH "Explicit OpenMM headers when OPENMM_ROOT is unset")
set(OPENMM_LIBRARY "" CACHE FILEPATH "Explicit OpenMM core library when OPENMM_ROOT is unset")
set(OPENMM_CUDA_LIBRARY "" CACHE FILEPATH "Explicit OpenMM CUDA library when OPENMM_ROOT is unset")

if(OPENMM_COLVARS_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()
if(OPENMM_COLVARS_BUILD_PYTHON AND NOT OPENMM_ROOT AND NOT OPENMM_INCLUDE_DIR AND NOT OPENMM_LIBRARY)
    # Building the SWIG wrapper only needs Python headers. An isolated build
    # Python need not yet have a working OpenMM runtime available to import.
    execute_process(COMMAND "${Python_EXECUTABLE}" -c
        "import importlib.util,pathlib,sys; pkg=importlib.util.find_spec('openmm'); print(pathlib.Path(pkg.origin).resolve().parent.parent/'OpenMM.libs' if pkg and pkg.origin else '', sys.prefix, sep=';')"
        OUTPUT_VARIABLE OPENMM_PYTHON_PREFIXES OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
endif()
if(OPENMM_ROOT)
    # An explicit SDK must take precedence over other activated environments,
    # including when this build directory previously used a different SDK.
    find_path(colvars_openmm_headers openmm/Context.h PATHS "${OPENMM_ROOT}"
        PATH_SUFFIXES include Library/include NO_DEFAULT_PATH NO_CACHE REQUIRED)
elseif(OPENMM_INCLUDE_DIR)
    set(colvars_openmm_headers "${OPENMM_INCLUDE_DIR}")
elseif(OPENMM_LIBRARY)
    # Handles prefix/lib, prefix/lib64 and prefix/lib/<architecture>.
    get_filename_component(sdk_lib_dir "${OPENMM_LIBRARY}" DIRECTORY)
    find_path(colvars_openmm_headers openmm/Context.h PATHS "${sdk_lib_dir}/.." "${sdk_lib_dir}/../.."
        PATH_SUFFIXES include NO_DEFAULT_PATH NO_CACHE REQUIRED)
else()
    if(OPENMM_PYTHON_PREFIXES)
        find_path(colvars_openmm_headers openmm/Context.h PATHS ${OPENMM_PYTHON_PREFIXES}
            PATH_SUFFIXES include Library/include NO_DEFAULT_PATH NO_CACHE)
    endif()
    if(NOT colvars_openmm_headers)
        find_path(colvars_openmm_headers openmm/Context.h HINTS "$ENV{CONDA_PREFIX}"
            PATH_SUFFIXES include Library/include NO_CACHE REQUIRED)
    endif()
endif()
set(OPENMM_INCLUDE_DIR "${colvars_openmm_headers}")
get_filename_component(openmm_sdk_prefix "${OPENMM_INCLUDE_DIR}" DIRECTORY)
if(OPENMM_ROOT OR NOT OPENMM_LIBRARY)
    find_library(colvars_openmm_library NAMES OpenMM PATHS "${openmm_sdk_prefix}"
        PATH_SUFFIXES lib lib64 "lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH NO_CACHE REQUIRED)
    set(OPENMM_LIBRARY "${colvars_openmm_library}")
elseif(NOT EXISTS "${OPENMM_LIBRARY}")
    message(FATAL_ERROR "OPENMM_LIBRARY does not exist: ${OPENMM_LIBRARY}")
endif()
get_filename_component(OPENMM_LIBRARY_DIR "${OPENMM_LIBRARY}" DIRECTORY)
set(openmm_cuda_library_dir "${OPENMM_LIBRARY_DIR}/plugins")
message(STATUS "OpenMM headers: ${OPENMM_INCLUDE_DIR}")
message(STATUS "OpenMM library: ${OPENMM_LIBRARY}")
if(NOT EXISTS "${OPENMM_INCLUDE_DIR}/openmm/internal/CustomCPPForceImpl.h")
    message(FATAL_ERROR "OpenMM with CustomCPPForceImpl is required (OpenMM 8.1 or newer)")
endif()

file(GLOB COLVARS_CORE_SOURCES CONFIGURE_DEPENDS "${COLVARS_CORE_DIR}/*.cpp")
add_library(OpenMMColvars STATIC
    src/ColvarsForce.cpp resident/src/ColvarsForceImpl.cpp src/ColvarsTABFIntegrator.cpp resident/src/ColvarsTABFEndpoint.cpp resident/src/colvarproxy_openmm.cpp
    ${COLVARS_CORE_SOURCES})
set_target_properties(OpenMMColvars PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(OpenMMColvars PUBLIC OPENMM_COLVARS_RESIDENT)
target_include_directories(OpenMMColvars PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/resident/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PRIVATE "${COLVARS_CORE_DIR}")
target_include_directories(OpenMMColvars SYSTEM PUBLIC "${OPENMM_INCLUDE_DIR}")
target_link_libraries(OpenMMColvars PUBLIC "${OPENMM_LIBRARY}")
if(WIN32 AND NOT CYGWIN)
    target_compile_definitions(OpenMMColvars PRIVATE COLVARS_UTF8_WINDOWS_PATHS)
endif()
if(MSVC)
    target_compile_options(OpenMMColvars PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus;/utf-8>")
else()
    target_compile_options(OpenMMColvars PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif()

# Reuse the matching OpenMM source's Lepton implementation when supplied.
set(LEPTON_DIR "" CACHE PATH "Optional explicit Lepton source directory")
if(LEPTON_DIR)
    set(COLVARS_LEPTON_DIR "${LEPTON_DIR}")
else()
    # Keep the inferred path out of the cache: users may supply or change
    # OPENMM_SOURCE_DIR on a later configure of the same build directory.
    set(COLVARS_LEPTON_DIR "${OPENMM_SOURCE_DIR}/libraries/lepton")
endif()
if(EXISTS "${COLVARS_LEPTON_DIR}/include/Lepton.h")
    file(GLOB LEPTON_SOURCES CONFIGURE_DEPENDS "${COLVARS_LEPTON_DIR}/src/*.cpp")
    target_sources(OpenMMColvars PRIVATE ${LEPTON_SOURCES})
    target_include_directories(OpenMMColvars SYSTEM PRIVATE "${COLVARS_LEPTON_DIR}/include")
    target_compile_definitions(OpenMMColvars PRIVATE LEPTON LEPTON_BUILDING_STATIC_LIBRARY)
    message(STATUS "Colvars customFunction support enabled with ${COLVARS_LEPTON_DIR}")
else()
    message(STATUS "Colvars customFunction disabled: supply OPENMM_SOURCE_DIR or LEPTON_DIR to enable")
endif()

if(OPENMM_COLVARS_ENABLE_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(OPENMM_ROOT OR NOT OPENMM_CUDA_LIBRARY)
        find_library(colvars_openmm_cuda_library NAMES OpenMMCUDA
            PATHS "${OPENMM_LIBRARY_DIR}/plugins" "${OPENMM_LIBRARY_DIR}"
            NO_DEFAULT_PATH NO_CACHE REQUIRED)
        set(OPENMM_CUDA_LIBRARY "${colvars_openmm_cuda_library}")
    elseif(NOT EXISTS "${OPENMM_CUDA_LIBRARY}")
        message(FATAL_ERROR "OPENMM_CUDA_LIBRARY does not exist: ${OPENMM_CUDA_LIBRARY}")
    endif()
    get_filename_component(openmm_cuda_library_dir "${OPENMM_CUDA_LIBRARY}" DIRECTORY)
    if(NOT EXISTS "${OPENMM_INCLUDE_DIR}/openmm/cuda/CudaContext.h")
        message(FATAL_ERROR "CUDA build requires installed OpenMM CUDA internal headers from the same OpenMM version")
    endif()
    target_sources(OpenMMColvars PRIVATE resident/src/CudaColvarsKernel.cpp resident/src/CudaTABFStep.cpp
        resident/src/CudaTabfForce.cpp resident/src/CudaTabfBias.cpp
        resident/src/CudaTabfJointAbf.cpp resident/src/CudaTabfJointMeta.cpp
        resident/src/CudaTabfDynamicsAdapter.cpp resident/src/CudaTabfVariables.cpp
        "${COLVARS_CORE_DIR}/cuda/colvaratoms_kernel.cu"
        "${COLVARS_CORE_DIR}/cuda/colvar_rotation_derivative_kernel.cu"
        "${COLVARS_CORE_DIR}/cuda/colvartypes_kernel.cu"
        "${COLVARS_CORE_DIR}/cuda/colvarcomp_distance_kernel.cu"
        "${COLVARS_CORE_DIR}/cuda/colvarcomp_coordnums_kernel.cu")
    target_compile_definitions(OpenMMColvars PUBLIC COLVARS_CUDA OPENMM_COLVARS_ENABLE_CUDA)
    target_include_directories(OpenMMColvars PRIVATE "${COLVARS_CORE_DIR}/cuda")
    target_link_libraries(OpenMMColvars PUBLIC CUDA::cudart "${OPENMM_CUDA_LIBRARY}")
    if(TARGET CUDA::nvtx3)
        target_link_libraries(OpenMMColvars PUBLIC CUDA::nvtx3)
    endif()
    set_property(TARGET OpenMMColvars PROPERTY CUDA_STANDARD 17)
    set_property(TARGET OpenMMColvars PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

if(OPENMM_COLVARS_BUILD_PYTHON)
    find_program(SWIG_EXECUTABLE swig REQUIRED)
    set(PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
    file(MAKE_DIRECTORY "${PYTHON_OUTPUT_DIR}")
    add_custom_command(OUTPUT "${PYTHON_OUTPUT_DIR}/openmm_colvars_wrap.cpp"
        "${PYTHON_OUTPUT_DIR}/openmm_colvars.py"
        COMMAND "${SWIG_EXECUTABLE}" -c++ -python
            -DOPENMM_COLVARS_RESIDENT "-I${CMAKE_CURRENT_SOURCE_DIR}/resident/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/src" "-I${CMAKE_CURRENT_SOURCE_DIR}/resident/python"
            -outdir "${PYTHON_OUTPUT_DIR}" -o "${PYTHON_OUTPUT_DIR}/openmm_colvars_wrap.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/resident/python/openmm_colvars.i"
        DEPENDS resident/python/openmm_colvars.i resident/python/openmm_colvars_openmm_import.i src/ColvarsForce.h src/ColvarsTABFIntegrator.h
        VERBATIM)
    Python_add_library(_openmm_colvars MODULE "${PYTHON_OUTPUT_DIR}/openmm_colvars_wrap.cpp")
    target_link_libraries(_openmm_colvars PRIVATE OpenMMColvars Python::Module)
    set_target_properties(_openmm_colvars PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_OUTPUT_DIR}"
        BUILD_RPATH "${OPENMM_LIBRARY_DIR};${openmm_cuda_library_dir}"
        INSTALL_RPATH "${OPENMM_LIBRARY_DIR};${openmm_cuda_library_dir}")
    if(NOT OPENMM_COLVARS_PYTHON_INSTALL_DIR)
        execute_process(COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
            OUTPUT_VARIABLE OPENMM_COLVARS_PYTHON_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
    install(TARGETS _openmm_colvars LIBRARY DESTINATION "${OPENMM_COLVARS_PYTHON_INSTALL_DIR}")
    install(FILES "${PYTHON_OUTPUT_DIR}/openmm_colvars.py" DESTINATION "${OPENMM_COLVARS_PYTHON_INSTALL_DIR}")
endif()
