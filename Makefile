cc        := /usr/bin/g++-9
name      := transform.so
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home :=
cuda_arch := 8.6
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)


project_include_path := src
opencv_include_path  := /usr/include/opencv4/
cuda_include_path    := $(cuda_home)/include



include_paths        := $(project_include_path) \
						$(opencv_include_path) \
						$(cuda_include_path)


opencv_library_path  := /workspace/__install/opencv490/lib
cuda_library_path    := $(cuda_home)/lib64/

library_paths        := $(opencv_library_path) \
						$(cuda_library_path) \
						$(cuda_library_path)

link_opencv       := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs
link_cuda         := cuda cublas cudart
link_sys          := stdc++ dl

link_librarys     := $(link_opencv) $(link_cuda) $(link_sys)


empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags_debug   := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread $(include_paths)
cpp_compile_flags_release := -std=$(stdcpp) -O2 -m64 -fPIC -fopenmp -pthread -march=native -Wall -Wextra $(include_paths)

cpp_compile_flags := $(cpp_compile_flags_release)

cu_compile_flags  := -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN' $(library_paths) $(link_librarys)

cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

pro_cpp_objs := $(filter-out objs/interface.cpp.o, $(cpp_objs))

ifneq ($(MAKECMDGOALS), clean)
include $(mks)
endif


$(name)   : $(workdir)/$(name)

all       : $(name)

pro       : $(workdir)/pro

run   : pro
	@cd $(workdir) && ./pro

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(workdir)/pro : $(pro_cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)

$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)


clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/pro $(workdir)/*.trtmodel $(workdir)/imgs

.PHONY : clean run $(name) runpro
