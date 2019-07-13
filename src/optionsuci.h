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
 * @file: optionsuci.h
 * Created on 13.07.2019
 * @author: queensgambit
 *
 * UCI Option definition and initialization with default values.
 */

#ifndef OPTIONSUCI_H
#define OPTIONSUCI_H

#include "uci.h"
#include "misc.h"
#include "domain/variants.h"

using namespace UCI;

namespace OptionsUCI {

//    void activate_logger(const Option& o) { start_logger(o); }

    /**
     * @brief init Defines and initiatlizes the UCI options
     * @param o Alias to the option map which will get initialized
     */
    void init(OptionsMap& o);
}

#endif // OPTIONSUCI_H
