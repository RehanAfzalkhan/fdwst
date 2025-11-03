% nonlinear cartoon/texture decomposition

# ABOUT

* Author    : Antoni Buades <toni.buades@gmail.com>
* Copyright : (C) 2009, 2010 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+, see GPLv3.txt

# OVERVIEW

This source code provides an implementation of the nonlinear
cartoon+texture decomposition, as described in IPOL
    http://www.ipol.im/pub/algo/blmv_nonlinear_cartoon_texture_decomposition/

This program reads and writes PNG images, but can be easily
adapted to any other file format.

Only 8bit RGB PNG images are handled. Other TIFF files are implicitly
converted to 8bit color RGB.

# REQUIREMENTS

The code is written in ANSI C and C++, and should compile on any
system with an ANSI C/C++ compiler.

The libpng header and libraries are required on the system for
compilation and execution. 

# COMPILATION

Simply use the provided makefile, with the command `make`.


# USAGE

`cartoonIpol` takes 4 parameters: `cartoonIpol in.png sigma cartoon.png texture.png`
* `sigma`     : the scale parameter
* `in.png`   : input image
* `cartoon.png`  : output image without the textures
* `textures.png` : textures (out - in, bounded to [-20, 20])

# ABOUT THIS FILE

Copyright 2009, 2010 IPOL Image Processing On Line http://www.ipol.im/
Author: Antoni Buades <toni.buades@gmail.com>

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.
