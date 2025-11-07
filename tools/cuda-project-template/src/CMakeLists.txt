# ************ README ************
# 1. 通常来说，在CMake中采用conan管理的依赖的包名为CONAN_PKG::<package_name>，
# 其中package_name为conanfile.txt中所指定的包名。
# 
# 2. 如何选择使用 PUBLIC 还是 PRIVATE 作为target系列函数的访问控制选项
#    - PUBLIC: 所链接的库应用在本lib的头文件中
#    - PRIVATE: 所链接的库只应用在本lib的源文件中
# ************ README ************

set(CMAKE_CXX_CPPLINT "cpplint;--filter=-build/c++11")

add_executable({{PROJECT_NAME}}
  main.cu
)

set_target_properties({{PROJECT_NAME}}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
