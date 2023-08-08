/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: nodedata.h
 * Created on 25.04.2020
 * @author: queensgambit
 *
 * Node data is a data container which is unavailable for all nodes <= 1 to reduce memory consumption.
 */

#ifndef NODEDATA_H
#define NODEDATA_H

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <blaze/Math.h>
#include "agents/config/searchsettings.h"

using blaze::HybridVector;
using blaze::DynamicVector;
using namespace std;


enum NodeType : uint8_t {
    WIN,
    DRAW,
    LOSS,
#ifdef MCTS_TB_SUPPORT
    TB_WIN,
    TB_DRAW,
    TB_LOSS,
#endif
    UNSOLVED
};

/**
 * @brief is_loss_node_type Returns true if given node type belongs to a loosing node type
 * @param nodeType
 * @return bool
 */
bool is_loss_node_type(NodeType nodeType);

/**
 * @brief is_win_node_type Returns true if given node type belongs to a win node type
 * @param nodeType
 * @return bool
 */
bool is_win_node_type(NodeType nodeType);

/**
 * @brief is_draw_node_type Returns true if given node type belongs to a drawubg node type
 * @param nodeType
 * @return  bool
 */
bool is_draw_node_type(NodeType nodeType);

/**
 * @brief is_unsolved_or_tablebase Checks if the given node type is a win, draw, loss or a different type.
 * @param nodeType given node type
 * @return bool
 */
bool is_unsolved_or_tablebase(NodeType nodeType);


class Node;

/**
 * @brief The NodeData struct stores the member variables for all expanded child nodes which have at least been visited two times
 */
struct NodeData
{
    DynamicVector<uint32_t> childNumberVisits;
    DynamicVector<float> qValues;
    vector<shared_ptr<Node>> childNodes;
    DynamicVector<uint8_t> virtualLossCounter;
    DynamicVector<NodeType> nodeTypes;

    uint32_t freeVisits;
    uint32_t visitSum;

    uint16_t checkmateIdx;
    uint16_t endInPly;
    uint16_t noVisitIdx;
    uint16_t numberUnsolvedChildNodes;

    NodeType nodeType;
    bool inspected;
    NodeData();
    NodeData(size_t numberChildNodes);

    auto get_q_values();

public:
    /**
     * @brief add_empty_node Adds a new empty node to its child nodes
     */
    void add_empty_node();

    /**
     * @brief reserve_initial_space Reserves memory for PRESERVED_ITEMS number of child nodes
     */
    void reserve_initial_space();
};


#endif // NODEDATA_H
