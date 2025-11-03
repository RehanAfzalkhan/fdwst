# Copyright 2010 Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
#		 Jose-Luis Lisani <joseluis.lisani@uib.es>
#
# Copying and distribution of this file, with or without
# modification, are permitted in any medium without royalty provided
# the copyright notice and this notice are preserved.  This file is
# offered as-is, without any warranty.


# C source code
CSRC	= io_png.c
# C++ source code
CXXSRC	= cartoonIpol.cpp libcartoon.cpp

# all source code
SRC	= $(CSRC) $(CXXSRC)

# C objects
COBJ	= $(CSRC:.c=.o)
# C++ objects
CXXOBJ	= $(CXXSRC:.cpp=.o)
# all objects
OBJ	= $(COBJ) $(CXXOBJ)
# binary target
BIN	= cartoonIpol

default	: $(BIN)


# C optimization flags
COPT	= -O3 -ftree-vectorize -funroll-loops \
         -fomit-frame-pointer  -ffast-math

# C++ optimization flags
CXXOPT	= $(COPT)

# C compilation flags
CFLAGS	= $(COPT) -Wall -Wextra -Werror \
	-Wno-write-strings -ansi
# C++ compilation flags
CXXFLAGS	= $(CXXOPT) -Wall -Wextra -Werror \
	-Wno-write-strings -Wno-deprecated -ansi \
        -Weffc++  -pedantic
# link flags
LDFLAGS	= -lpng 




# partial compilation of C source code
%.o: %.c %.h
	$(CC) -I/opt/local/include/ -I/usr/local/include/ -c -o $@  $< $(CFLAGS)
# partial compilation of C++ source code
%.o: %.cpp %.h
	$(CXX) -I/opt/local/include/ -I/usr/local/include/ -c -o $@  $< $(CXXFLAGS)

# link all the opject code
$(BIN): $(OBJ) $(LIBDEPS)
	$(CXX)  -I/opt/local/include/ -I/usr/local/include/   -L/opt/local/lib/ -L/usr/local/lib/     -o $@ $(OBJ) $(LDFLAGS)

# housekeeping
.PHONY	: clean distclean
clean	:
	$(RM) $(OBJ)
distclean	: clean
	$(RM) $(BIN)

