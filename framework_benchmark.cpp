/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: framework_benchmark.cpp
 * Created on 19.07.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

//#define BENCHMARK
#ifdef BENCHMARK

//#define EIGEN_USE_MKL_ALL

#include <iostream>

// MXNet
#include "mxnet-cpp/MxNetCpp.h"
using namespace mxnet::cpp;

// Xtensor
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xinfo.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xsort.hpp"

// Eigen
#include <Dense>
using Eigen::MatrixXd;

// blaze
#include <blaze/Math.h>
using blaze::StaticVector;
using blaze::DynamicVector;

using namespace std;


int main() {
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    cout << "Script start" << endl;

//    blaze::DynamicVector<int> v1( sizeUL ), v3;

    // ... Initializing the vectors
    const int size = 80; //34;

    NDArray mxnet_vec = NDArray(Shape(size, 1), Context::cpu());
//    auto xtensor_vec =  xt::ones<float>({size}); //xt::random::rand<float>({size}); //
    xt::xarray<float> xtensor_vec{1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9,
                                  1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9};

//    blaze::HybridVector<float, 1024UL> blaze_vec(size);
//    blaze::UniformVector<float> blaze_vec(sizeUL);
//    const float data[size] = {mxnet_vec.GetData()};
//    std::vector<float> data(mxnet_vec.GetData(), size);
//    const float *data = mxnet_vec.GetData();
//    float result2 = *dat; // works!!

//    data = &mxnet_vec.GetData()[0];
//    blaze_vec = data; //mxnet_vec.GetData(); //99.9f;
    auto eigen_vec =  Eigen::VectorXd(size); //::Ones(size); //Eigen::VectorXd::Random(size); //
    eigen_vec << 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9,
                 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9;
    float vec[size];
    auto eigen_vec_div =  Eigen::VectorXd(size); //::Ones(size); //Eigen::VectorXd::Random(size); //
    eigen_vec_div << 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/
            1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9;

    float cur_max = eigen_vec[0];
    int max_idx = 0;
    for (int i = 0; i < size; ++i) {
        vec[i] = eigen_vec[i];
        if (vec[i] > cur_max) {
            max_idx = i;
            cur_max = vec[i];
        }
    }
    cout << "correct max_idx" << max_idx << endl;

//    cout << "eigen_vec" << eigen_vec << endl;
//    cout << "vec" << endl;
    for (auto i : vec) {
//        cout << i << " ";
    }
//    cout << eigen_vec << endl;
//    vector<int> vec(eigen_vec.data(), eigen_vec.data() + mat.cols());
//    blaze::StaticVector<float, size> blaze_vec;
//    blaze::HybridVector<float, 512UL> blaze_vec(size);
//    blaze::DynamicVector<float> blaze_vec(size);
    typedef double real;
//    typedef float real;

     blaze::DynamicVector<real> blaze_vec {1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9,
                                           1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9};
//    blaze_vec = vec;
//    cout << "blaze vec" << blaze_vec << endl;

//    auto res_xtensor = xt::random::rand<float>({size});
    NDArray res_mxnet = NDArray(Shape(size, 1), Context::cpu());
    xt::xarray<float> res_xtensor = xt::random::rand<float>({size});
//    for (int i = 0; i < size; ++i) {
//        res_xtensor[i] = vec[i];
//    }

//    xtensor_vec[42] = 99;
//    blaze_vec[42] = 99;
//    eigen_vec[42] = 99;

//    blaze::StaticVector<float, size> res_blaze;
    blaze::DynamicVector<real> res_blaze(size);
//    blaze::HybridVector<float, 512UL> res_blaze(size);

    res_blaze = 0;
    Eigen::VectorXd res_eigen = Eigen::VectorXd::Zero(size);  //Random(size);

    size_t it = 1e7; //7; //999999;

    std::chrono::steady_clock::time_point start_mxnet = std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
//        res_mxnet += mxnet_vec;
//        res_mxnet *= mxnet_vec;
//        res_mxnet.WaitToWrite();
//        res_mxnet /= mxnet_vec;
//        res_mxnet.WaitToWrite();
    }
    std::chrono::steady_clock::time_point end_mxnet = std::chrono::steady_clock::now();
    std::cout << "Elapsed time mxnet:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_mxnet - start_mxnet).count() << "ms" << std::endl;

//    int id = 0;
    auto id = xt::argmax(xtensor_vec);

    std::chrono::steady_clock::time_point start= std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
//        auto sum_xtensor = xt::sum(xtensor_vec);
//          res_eigen = xtensor_vec + xtensor_vec;
            res_xtensor += xtensor_vec;
//            res_xtensor *= xtensor_vec;
//            res_xtensor /= xtensor_vec;
//            id = xt::argmax(xtensor_vec);
    }
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Elapsed time xtensor:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    int idx = 0;
    std::chrono::steady_clock::time_point start_blaze = std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
        res_blaze += blaze_vec;
//        res_blaze *= blaze_vec;
//        res_blaze /= blaze_vec;
////        argmax(blaze_vec);
//        res_blaze = sqrt(res_blaze);
//         idx = argmax(blaze_vec); //blaze_vec.find(max(blaze_vec));
    }

    std::chrono::steady_clock::time_point end_blaze = std::chrono::steady_clock::now();
    std::cout << "Elapsed time blaze:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_blaze - start_blaze).count() << "ms" << std::endl;

    int index = 0;
    const float totalsum = sum( res_blaze );  // Results in 10
//    cout << "last number" << vec[511] * it << endl;
//    cout << "totalsum" << res_blaze[511] << endl;
    std::chrono::steady_clock::time_point start_eigen = std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
        res_eigen += eigen_vec;
//        res_eigen *= eigen_vec;
//        res_eigen *= eigen_vec_div;
//        res_eigen *= (1.0f/eigen_vec);
//        eigen_vec.maxCoeff(&index);
    }
//    cout << "res einge" << res_eigen[511] << endl;


    std::chrono::steady_clock::time_point end_eigen = std::chrono::steady_clock::now();
    std::cout << "Elapsed time eigen:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_eigen - start_eigen).count() << "ms" << std::endl;

//    cout << "argmax xtensor " << id << endl;
//    cout << "argmax blaze_vec " << idx << endl;
//    cout << "argmax eigen_vec " << index << endl;

//        cout << "res mxnet " << res_mxnet.At(0,7) << endl;
//        cout << "res xtensor " << res_xtensor[7] << endl;
//        cout << "res blaze_vec " << res_blaze[7] << endl;
//        cout << "res eigen_vec " << res_eigen[7] << endl;

    // -> blaze is faster than all competitors for this matrix size by far!

    cout << endl;
}
#endif
