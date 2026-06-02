# csdk/Makefile — build reference C controllers as shared libraries.
#
# Targets:
#   make                         build all examples for this platform
#   make examples/libpid_hover   build a specific example
#   make clean
#
# Platform detection: macOS produces .dylib; Linux produces .so.
# Override at the command line if cross-compiling: `make EXT=.so`.

CC      ?= cc
CFLAGS  ?= -O2 -fPIC -Wall -Wextra -std=c11
INCLUDES = -Iinclude

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    EXT        ?= .dylib
    SHARED_FLAG = -dynamiclib
else
    EXT        ?= .so
    SHARED_FLAG = -shared
endif

EXAMPLES = pid_hover

.PHONY: all clean check_header

all: $(foreach ex,$(EXAMPLES),examples/lib$(ex)$(EXT))

examples/lib%$(EXT): examples/%.c include/simulacra.h
	$(CC) $(CFLAGS) $(INCLUDES) $(SHARED_FLAG) -o $@ $<

clean:
	rm -f examples/lib*$(EXT)
	rm -f examples/*.o

# Sanity check: the header compiles standalone.
check_header:
	$(CC) -fsyntax-only -Iinclude -xc include/simulacra.h
