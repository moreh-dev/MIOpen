# vars
MIOPEN_INTERNAL_INC_PATH=/home/jungwook/MIOpen_moreh/MIOpen/src/include
MIOPEN_INC_PATH=/home/jungwook/MIOpen_moreh/MIOpen/include
ROCM_INC_PATH=/opt/rocm/include

LIB_MIOPEN_PATH=/home/jungwook/MIOpen_moreh/moreh/lib/libMIOpen.so.1.0

CC := /opt/rocm/llvm/bin/clang++


# rules
CPPFLAGS := -D__HIP_PLATFORM_AMD__=1

OBJS := main.o InputFlags.o

miopen_db : $(OBJS)
	$(CC) -I$(ROCM_INC_PATH) $(LIB_MIOPEN_PATH) $(OBJS) 

main.o : main.cpp
	$(CC) -I$(MIOPEN_INTERNAL_INC_PATH) -I$(MIOPEN_INC_PATH) -I$(ROCM_INC_PATH) $(CPPFLAGS) -c $< -o $@

InputFlags.o : InputFlags.cpp
	$(CC) -I$(MIOPEN_INTERNAL_INC_PATH) -I$(MIOPEN_INC_PATH) -I$(ROCM_INC_PATH) $(CPPFLAGS) -c $< -o $@

clean:
	rm *.o a.out
