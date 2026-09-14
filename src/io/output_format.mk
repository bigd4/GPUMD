# Included by the CUDA and HIP makefiles; no network access is needed.
BXYZ_LIBS ?= -lzstd -lcrypto
BXYZ_CXXFLAGS ?=
USE_BXYZ ?= 0
BXYZ_ROOT = third_party/binary_xyz
ifeq ($(USE_BXYZ),1)
OUTPUT_OBJ = io/output_format_bxyz.o io/binary_xyz.o
OUTPUT_LIBS = $(BXYZ_LIBS)
else
OUTPUT_OBJ = io/output_format_text.o
OUTPUT_LIBS =
endif
# The adapter is the only translation unit that sees the C++17 codec API.
io/output_format_text.o: io/output_format.cpp io/output_format.h
	$(CXX) -std=c++14 -O3 -I./ -c $< -o $@
io/output_format_bxyz.o: io/output_format.cpp io/output_format.h $(BXYZ_ROOT)/cpp/include/binary_xyz/binary_xyz.hpp
	$(CXX) $(BXYZ_CXXFLAGS) -std=c++17 -O3 -DUSE_BXYZ -I./ -I$(BXYZ_ROOT)/cpp/include -c $< -o $@
io/binary_xyz.o: $(BXYZ_ROOT)/cpp/src/binary_xyz.cpp $(BXYZ_ROOT)/cpp/include/binary_xyz/binary_xyz.hpp
	$(CXX) $(BXYZ_CXXFLAGS) -std=c++17 -O3 -I$(BXYZ_ROOT)/cpp/include -c $< -o $@

# Switching USE_BXYZ must relink even when both adapter objects already exist.
.PHONY: FORCE_OUTPUT_MODE
io/.output_mode: FORCE_OUTPUT_MODE
	@printf '%s\n' '$(USE_BXYZ)' | cmp -s - $@ || printf '%s\n' '$(USE_BXYZ)' > $@
