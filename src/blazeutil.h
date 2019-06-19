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
 * @file: blazeutil.h
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef BLAZEUTIL_H
#define BLAZEUTIL_H

#include <blaze/Math.h>

//#include <blaze/math/expressions/DenseVector.h>
//#include <blaze/math/expressions/Vector.h>

using blaze::StaticVector;
using blaze::DynamicVector;


template< typename VT  // Type of the vector
        , bool TF >    // Transpose flag
struct Vector
{
   //**Type definitions****************************************************************************
   using VectorType = VT;  //!< Type of the vector.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   static constexpr bool transposeFlag = TF;  //!< Transpose flag of the vector.
   //**********************************************************************************************

   //**Non-const conversion operator***************************************************************
   /*!\brief Conversion operator for non-constant vectors.
   //
   // \return Reference of the actual type of the vector.
   */
   BLAZE_ALWAYS_INLINE constexpr VectorType& operator~() noexcept {
      return *static_cast<VectorType*>( this );
   }
   //**********************************************************************************************

   //**Const conversion operators******************************************************************
   /*!\brief Conversion operator for constant vectors.
   //
   // \return Const reference of the actual type of the vector.
   */
   BLAZE_ALWAYS_INLINE constexpr const VectorType& operator~() const noexcept {
      return *static_cast<const VectorType*>( this );
   }
   //**********************************************************************************************
};

template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
struct DenseVector
   : public Vector<VT,TF>
{};


template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline decltype(auto) argmax( const DynamicVector<VT,TF>& dv ) {
    size_t idx = 0;      // set the first index as the hypothesis for the armgax index
    VT cur_max = dv[0];  // initialize the current maximum value

    // iterate through the vector and update idx and cur_max accordingly
    for( size_t i=1UL; i<dv.size(); ++i ) {
        if (dv[i] > cur_max) {
            idx = i;
            cur_max = dv[i];
        }
    }
    return idx;
}


#endif // BLAZEUTIL_H
