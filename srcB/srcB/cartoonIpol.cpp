/*
 * Copyright (c) 2009-2011, A. Buades <toni.buades@uib.es>
 * All rights reserved.
 *  
 *
 * 
 * License:
 *
 * This program is provided for scientific and educational only:
 * you can use and/or modify it for these purposes, but you are
 * not allowed to redistribute this work or derivative works in
 * source or executable form. A license must be obtained from the
 * patent right holders for any other use.
 *
 *
 */

/**
 * @mainpage Nonlinear Cartoon + Texture decomposition
 *
 * README.txt:
 * @verbinclude README.txt
 */



/**
 * @file   cartoonIpol.cpp
 * @brief  Main executable file
 *
 * 
 *
 * @author Antoni Buades <toni.buades@uib.es>
 */





#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "libcartoon.h"
#include "io_png.h"



int main(int argc, char **argv)
{
        
        
        if (argc < 5)
        {
                printf("usage: cartoonIpol image sigma cartoon texture \n");
                return EXIT_FAILURE;
        }
        
        
        
        
        // read input
        size_t nx,ny,nc;
        float *d_v = NULL;
        d_v = io_png_read_f32(argv[1], &nx, &ny, &nc);
        if (!d_v)
        {
           printf("error :: %s not found  or not a correct png image \n", 
                   argv[1]);
           return EXIT_FAILURE;
        }
        
        
        
        
        // variables
        int d_w = (int) nx; 
        int d_h = (int) ny; 
        int d_c = (int) nc; 
        if (d_c == 2) { d_c = 1; }  // we do not use the alpha channel
        if (d_c > 3) { d_c = 3; }  // we do not use the alpha channel
        
        int d_wh = d_w * d_h;
        int d_whc = d_c * d_w * d_h;
        
        
        
        
        // parameters : input 
        float fSigma = atof(argv[2]);
        
        
        

        
        
        
        
        
        // build cartoon
        float *cartoon_v = new float[d_c * d_wh];

        if (d_c == 1)
        {
           non_linear_cartoon(d_v, cartoon_v, fSigma,  d_w,  d_h);
                
        } else if (d_c == 3)
        {
           non_linear_cartoon(d_v, &d_v[d_wh], &d_v[2*d_wh],
                              cartoon_v, &cartoon_v[d_wh], &cartoon_v[2*d_wh],
                              fSigma, d_w,  d_h);

        }


        
        // compute texture as difference as transform from [-20,20] to [0,255]
        float vLim = 20;
        float *dif_v = new float[d_c * d_wh];
        for (int ii=0; ii < d_whc; ii++)
        {
                
                float fValue = d_v[ii] - cartoon_v[ii];
                fValue =  (fValue + vLim) * 255.0f / (2.0f * vLim);
        
                if (fValue < 0.0) fValue = 0.0f;
                if (fValue > 255.0) fValue = 255.0f;
        
                
                dif_v[ii] = fValue;
        }
        
        
        
        // save cartoon and texture images
        if (io_png_write_f32(argv[3], cartoon_v, (size_t) d_w, (size_t) d_h,
                             (size_t) d_c) != 0)
        {
                printf("... failed to save png image %s", argv[3]);
        } 
        
        if (io_png_write_f32(argv[4], dif_v, (size_t) d_w, (size_t) d_h, 
                             (size_t) d_c) != 0)
        {
                printf("... failed to save png image %s", argv[4]);
        } 
        
        
	delete[] cartoon_v;
	delete[] dif_v;
	
	return EXIT_SUCCESS;
        
}
