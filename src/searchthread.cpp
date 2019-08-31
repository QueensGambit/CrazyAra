/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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
 * @file: searchthread.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "searchthread.h"
#include "inputrepresentation.h"
#include "outputrepresentation.h"
#include "uci.h"

SearchThread::SearchThread(NeuralNetAPI *netBatch, SearchSettings* searchSettings, unordered_map<Key, Node *> *hashTable):
    netBatch(netBatch), isRunning(false), hashTable(hashTable), searchSettings(searchSettings)
{
    // allocate memory for all predictions and results
    inputPlanes = new float[searchSettings->batchSize * NB_VALUES_TOTAL];
    valueOutputs = new NDArray(Shape(searchSettings->batchSize, 1), Context::cpu());

    bool select_policy_from_plane = true;

    if (select_policy_from_plane) {
        probOutputs = new NDArray(Shape(searchSettings->batchSize, NB_LABELS_POLICY_MAP), Context::cpu());
    } else {
        probOutputs = new NDArray(Shape(searchSettings->batchSize, NB_LABELS), Context::cpu());
    }
}

void SearchThread::set_root_node(Node *value)
{
    rootNode = value;
}

void SearchThread::set_search_limits(SearchLimits *s)
{
    searchLimits = s;
}

bool SearchThread::get_is_running() const
{
    return isRunning;
}

void SearchThread::set_is_running(bool value)
{
    isRunning = value;
}

void SearchThread::stop()
{
    isRunning = false;
}

Node *SearchThread::get_root_node() const
{
    return rootNode;
}

SearchLimits *SearchThread::get_search_limits() const
{
    return searchLimits;
}

inline Node* get_new_child_to_evaluate(Node* rootNode, bool &isCollision, bool &isTerminal, size_t &depth)
{
    Node *currentNode = rootNode;
    rootNode->apply_virtual_loss();
    depth = 0;
    while (true) {
        currentNode = select_child_node(currentNode);
        currentNode->apply_virtual_loss();
        depth++;
        if (!currentNode->is_expanded()) {
            currentNode->init_board();
            currentNode->expand();
            isCollision = false;
            isTerminal = currentNode->is_terminal();
            return currentNode;
        }
        if (currentNode->is_terminal()) {
            isCollision = false;
            isTerminal = true;
            return currentNode;
        }
        if (!currentNode->has_nn_results()) {
            isCollision = true;
            isTerminal = false;
            return currentNode;
        }
    }
}

void SearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: newNodes) {
        if (!node->is_terminal()) {
            fill_nn_results(batchIdx, netBatch->is_policy_map(), searchSettings, valueOutputs, probOutputs, node);
        }
        ++batchIdx;
        hashTable->insert({node->get_pos()->hash_key(), node});
    }
}

void SearchThread::backup_value_outputs()
{
    backup_values(newNodes);
    backup_values(transpositionNodes);
    backup_values(collisionNodes);
}

void SearchThread::backup_collisions()
{
    for (auto node: collisionNodes) {
        backup_collision(node);
    }
    collisionNodes.clear();
}

void SearchThread::copy_node(const unordered_map<Key,Node*>::const_iterator &it, Board* newPos, Node* parentNode, size_t childIdx)
{
//    Node *newNode = new Node(*it->second);
//    newNode->mtx.lock();
//    newNode->pos = newPos;
//    newNode->parentNode = parentNode;
//    newNode->childIdxForParent = childIdx;
//    newNode->hasNNResults = true;
//    newNode->mtx.unlock();
//    parentNode->mtx.lock();
//    parentNode->childNodes[childIdx] = newNode;
//    parentNode->mtx.unlock();
}

bool SearchThread::nodes_limits_ok()
{
    return searchLimits->nodes == 0 || (rootNode->get_visits() < searchLimits->nodes);
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to the mini-batch
    Node *currentNode;
    unsigned int childIdx;
    bool isCollision;
    bool isTerminal;

    size_t depth;
    size_t numberNewNodes = 0;
    size_t tranpositionEvents = 0;
    size_t terminalEvents = 0;

    while (newNodes.size() < searchSettings->batchSize &&
           collisionNodes.size() < searchSettings->batchSize &&
           tranpositionEvents < searchSettings->batchSize &&
           terminalEvents < searchSettings->batchSize) {
        currentNode = get_new_child_to_evaluate(rootNode, isCollision, isTerminal, depth);

        if(isTerminal) {
            //            terminalNodes.push_back(parentNode->childNodes[childIdx]);
            backup_value(currentNode, currentNode->get_value());
            ++terminalEvents;
        }
        else if (isCollision) {
            // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
            collisionNodes.push_back(currentNode);
        }
        else {
            auto it = hashTable->find(currentNode->hash_key());
//            if(searchSettings->useTranspositionTable && it != hashTable->end() && it->second->hasNNResults &&
//                    it->second->pos->getStateInfo()->pliesFromNull == newState->pliesFromNull &&
//                    it->second->pos->getStateInfo()->rule50 == newState->rule50 &&
//                    newState->repetition == 0)
//            {
//                copy_node(it, newPos, currentNode, childIdx);
//                transpositionNodes.push_back(currentNode->childNodes[childIdx]);
////                parentNode->backup_value(childIdx, -parentNode->childNodes[childIdx]->value);
//                ++tranpositionEvents;
//            }
//            else {
                prepare_node_for_nn(currentNode, numberNewNodes, newNodes, inputPlanes);
                ++numberNewNodes;
//            }
        }
    }

}

void SearchThread::thread_iteration()
{
    create_mini_batch();
    if (newNodes.size() > 0) {
        netBatch->predict(inputPlanes, *valueOutputs, *probOutputs);
        set_nn_results_to_child_nodes();
    }
//    cout << "backup values" << endl;
    backup_value_outputs();
    backup_collisions();
//    rootNode->numberVisits = sum(rootNode->childNumberVisits);
}

void go(SearchThread *t)
{
    t->set_is_running(true);

    do {
        t->thread_iteration();
    } while(t->get_is_running() && t->nodes_limits_ok());
}

void backup_values(vector<Node*>& nodes)
{
    for (auto node: nodes) {
        backup_value(node, node->get_value());
    }
    nodes.clear();
}

void prepare_node_for_nn(Node* newNode,  size_t numberNewNodes, vector<Node*>& newNodes, float* inputPlanes)
{
    // save a reference newly created list in the temporary list for node creation
    // it will later be updated with the evaluation of the NN
    newNodes.push_back(newNode);

    // fill a new board in the input_planes vector
    // we shift the index by NB_VALUES_TOTAL each time
    board_to_planes(newNode->get_pos(), newNode->get_pos()->getStateInfo()->repetition, true, inputPlanes+numberNewNodes*NB_VALUES_TOTAL);
}

void fill_nn_results(size_t batchIdx, bool is_policy_map, const SearchSettings* searchSettings, NDArray* valueOutputs, NDArray* probOutputs, Node *node)
{
    vector<Move> legalMoves = retrieve_legal_moves(node->get_child_nodes());
    DynamicVector<float> policyProbSmall(legalMoves.size());
    get_probs_of_moves(get_policy_data_batch(batchIdx, probOutputs, is_policy_map),
                       legalMoves,
                       get_current_move_lookup(node->side_to_move()),
                       policyProbSmall);
    if (!is_policy_map) {
        apply_softmax(policyProbSmall);
    }
    enhance_moves(searchSettings, node->get_pos(), legalMoves, policyProbSmall);
    node->set_nn_results(valueOutputs->At(batchIdx, 0), policyProbSmall);
}
