# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/jyoti/data/Ego_class_Exp/optical_flow/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/jyoti/data/Ego_class_Exp/optical_flow/build

# Include any dependencies generated for this target.
include CMakeFiles/OF.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/OF.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OF.dir/flags.make

CMakeFiles/OF.dir/main.cpp.o: CMakeFiles/OF.dir/flags.make
CMakeFiles/OF.dir/main.cpp.o: /media/jyoti/data/Ego_class_Exp/optical_flow/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/jyoti/data/Ego_class_Exp/optical_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/OF.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OF.dir/main.cpp.o -c /media/jyoti/data/Ego_class_Exp/optical_flow/src/main.cpp

CMakeFiles/OF.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OF.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/jyoti/data/Ego_class_Exp/optical_flow/src/main.cpp > CMakeFiles/OF.dir/main.cpp.i

CMakeFiles/OF.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OF.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/jyoti/data/Ego_class_Exp/optical_flow/src/main.cpp -o CMakeFiles/OF.dir/main.cpp.s

CMakeFiles/OF.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/OF.dir/main.cpp.o.requires

CMakeFiles/OF.dir/main.cpp.o.provides: CMakeFiles/OF.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/OF.dir/build.make CMakeFiles/OF.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/OF.dir/main.cpp.o.provides

CMakeFiles/OF.dir/main.cpp.o.provides.build: CMakeFiles/OF.dir/main.cpp.o


CMakeFiles/OF.dir/utils.cpp.o: CMakeFiles/OF.dir/flags.make
CMakeFiles/OF.dir/utils.cpp.o: /media/jyoti/data/Ego_class_Exp/optical_flow/src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/jyoti/data/Ego_class_Exp/optical_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/OF.dir/utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OF.dir/utils.cpp.o -c /media/jyoti/data/Ego_class_Exp/optical_flow/src/utils.cpp

CMakeFiles/OF.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OF.dir/utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/jyoti/data/Ego_class_Exp/optical_flow/src/utils.cpp > CMakeFiles/OF.dir/utils.cpp.i

CMakeFiles/OF.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OF.dir/utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/jyoti/data/Ego_class_Exp/optical_flow/src/utils.cpp -o CMakeFiles/OF.dir/utils.cpp.s

CMakeFiles/OF.dir/utils.cpp.o.requires:

.PHONY : CMakeFiles/OF.dir/utils.cpp.o.requires

CMakeFiles/OF.dir/utils.cpp.o.provides: CMakeFiles/OF.dir/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/OF.dir/build.make CMakeFiles/OF.dir/utils.cpp.o.provides.build
.PHONY : CMakeFiles/OF.dir/utils.cpp.o.provides

CMakeFiles/OF.dir/utils.cpp.o.provides.build: CMakeFiles/OF.dir/utils.cpp.o


# Object files for target OF
OF_OBJECTS = \
"CMakeFiles/OF.dir/main.cpp.o" \
"CMakeFiles/OF.dir/utils.cpp.o"

# External object files for target OF
OF_EXTERNAL_OBJECTS =

OF: CMakeFiles/OF.dir/main.cpp.o
OF: CMakeFiles/OF.dir/utils.cpp.o
OF: CMakeFiles/OF.dir/build.make
OF: /usr/local/lib/libopencv_cudabgsegm.so.3.2.0
OF: /usr/local/lib/libopencv_cudaobjdetect.so.3.2.0
OF: /usr/local/lib/libopencv_cudastereo.so.3.2.0
OF: /usr/local/lib/libopencv_shape.so.3.2.0
OF: /usr/local/lib/libopencv_stitching.so.3.2.0
OF: /usr/local/lib/libopencv_superres.so.3.2.0
OF: /usr/local/lib/libopencv_videostab.so.3.2.0
OF: /usr/local/lib/libopencv_cudafeatures2d.so.3.2.0
OF: /usr/local/lib/libopencv_cudacodec.so.3.2.0
OF: /usr/local/lib/libopencv_cudaoptflow.so.3.2.0
OF: /usr/local/lib/libopencv_cudalegacy.so.3.2.0
OF: /usr/local/lib/libopencv_calib3d.so.3.2.0
OF: /usr/local/lib/libopencv_cudawarping.so.3.2.0
OF: /usr/local/lib/libopencv_features2d.so.3.2.0
OF: /usr/local/lib/libopencv_flann.so.3.2.0
OF: /usr/local/lib/libopencv_objdetect.so.3.2.0
OF: /usr/local/lib/libopencv_highgui.so.3.2.0
OF: /usr/local/lib/libopencv_ml.so.3.2.0
OF: /usr/local/lib/libopencv_photo.so.3.2.0
OF: /usr/local/lib/libopencv_cudaimgproc.so.3.2.0
OF: /usr/local/lib/libopencv_cudafilters.so.3.2.0
OF: /usr/local/lib/libopencv_cudaarithm.so.3.2.0
OF: /usr/local/lib/libopencv_video.so.3.2.0
OF: /usr/local/lib/libopencv_videoio.so.3.2.0
OF: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
OF: /usr/local/lib/libopencv_imgproc.so.3.2.0
OF: /usr/local/lib/libopencv_core.so.3.2.0
OF: /usr/local/lib/libopencv_cudev.so.3.2.0
OF: CMakeFiles/OF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/jyoti/data/Ego_class_Exp/optical_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable OF"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/OF.dir/build: OF

.PHONY : CMakeFiles/OF.dir/build

CMakeFiles/OF.dir/requires: CMakeFiles/OF.dir/main.cpp.o.requires
CMakeFiles/OF.dir/requires: CMakeFiles/OF.dir/utils.cpp.o.requires

.PHONY : CMakeFiles/OF.dir/requires

CMakeFiles/OF.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OF.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OF.dir/clean

CMakeFiles/OF.dir/depend:
	cd /media/jyoti/data/Ego_class_Exp/optical_flow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/jyoti/data/Ego_class_Exp/optical_flow/src /media/jyoti/data/Ego_class_Exp/optical_flow/src /media/jyoti/data/Ego_class_Exp/optical_flow/build /media/jyoti/data/Ego_class_Exp/optical_flow/build /media/jyoti/data/Ego_class_Exp/optical_flow/build/CMakeFiles/OF.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/OF.dir/depend

