/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>
#include <cstdio>

#include "driver.hpp"
#include "conv_driver.hpp"

int main() {

    ConvDriver<float, float>* drv = new ConvDriver<float, float>();
    drv->Pre();

    // FIXME
    //std::string filename = "/data/MIOpen_moreh/MIOpen/src/kernels/gfx90a68.HIP.fdb.txt";
    std::string filename = "/data/MIOpen_moreh/MIOpen/build_ocl/test.fdb.txt";

    std::ifstream file(filename);

    if(!file) {
        std::cerr << "find-db file not readable: " << filename << std::endl;
        return 0;
    }

    int n_line = 0;
    while(true)
    {
        std::string line;
        if(!std::getline(file, line))
            break;
        ++n_line;
        const auto next_line_begin = file.tellg();

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
        if(!is_key)
        {
            if(!line.empty()) 
            {
                std::cerr << "Ill-formed record: key not found: " << filename << "#" << n_line << std::endl;
            }
            continue;
        }
        const auto current_key = line.substr(0, key_size);

        drv->ParseKey(current_key);
        drv->Post();

        //break;        
    }


}