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

SearchThread::SearchThread(NeuralNetAPI *netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex):
    netBatch(netBatch), isRunning(false), mapWithMutex(mapWithMutex), searchSettings(searchSettings)
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
    searchLimits = nullptr;  // will be set by set_search_limits() every time before go()
}

SearchThread::~SearchThread()
{
    delete [] inputPlanes;
    delete valueOutputs;
    delete probOutputs;
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

bool is_transposition_verified(const unordered_map<Key,Node*>::const_iterator& it, const StateInfo* stateInfo) {
    return  it->second->has_nn_results() &&
            it->second->get_pos()->get_state_info()->pliesFromNull == stateInfo->pliesFromNull &&
            it->second->get_pos()->get_state_info()->rule50 == stateInfo->rule50 &&
            stateInfo->repetition == 0;
}

Node* get_new_child_to_evaluate(Node* rootNode, bool useTranspositionTable, MapWithMutex* mapWithMutex, NodeDescription& description)
{
    Node* parentNode = rootNode;
    Node* currentNode;
    rootNode->apply_virtual_loss();
    description.depth = 0;
    while (true) {
        currentNode = select_child_node(parentNode);
        currentNode->lock();
        currentNode->apply_virtual_loss();
        description.depth++;
        if (!currentNode->is_expanded()) {
            currentNode->init_board();
            mapWithMutex->mtx.lock();
            unordered_map<Key, Node*>::const_iterator it = mapWithMutex->hashTable->find(currentNode->hash_key());
            mapWithMutex->mtx.unlock();
            if(useTranspositionTable && it != mapWithMutex->hashTable->end() &&
                    is_transposition_verified(it, currentNode->get_pos()->get_state_info())) {
                *currentNode = *it->second;  // call of assignment operator
                currentNode->set_parent_node(parentNode);
                parentNode->increment_no_visit_idx();
                description.isCollision = false;
                description.isTerminal = currentNode->is_terminal();
                description.isTranposition = true;
                currentNode->unlock();
                return currentNode;
            }
            else {
                currentNode->expand();
                description.isCollision = false;
                description.isTerminal = currentNode->is_terminal();
                description.isTranposition = false;
                currentNode->unlock();
                return currentNode;
            }
        }
        if (currentNode->is_terminal()) {
            description.isCollision = false;
            description.isTerminal = true;
            description.isTranposition = false;
            currentNode->unlock();
            return currentNode;
        }
        if (!currentNode->has_nn_results()) {
            description.isCollision = true;
            description.isTerminal = false;
            description.isTranposition = false;
            currentNode->unlock();
            return currentNode;
        }
        currentNode->unlock();
        parentNode = currentNode;
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
        mapWithMutex->mtx.lock();
        mapWithMutex->hashTable->insert({node->get_pos()->hash_key(), node});
        mapWithMutex->mtx.unlock();
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

bool SearchThread::nodes_limits_ok()
{
    return searchLimits->nodes == 0 || (rootNode->get_visits() < searchLimits->nodes);
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to the mini-batch
    Node *currentNode;
    NodeDescription description;

    while (newNodes.size() < searchSettings->batchSize &&
           collisionNodes.size() < searchSettings->batchSize &&
           transpositionNodes.size() < searchSettings->batchSize &&
           terminalNodes.size() < searchSettings->batchSize) {
        currentNode = get_new_child_to_evaluate(rootNode, searchSettings->useTranspositionTable, mapWithMutex, description);

        if (description.isTranposition) {
            transpositionNodes.push_back(currentNode);
        }
        else if(description.isTerminal) {
            //                        terminalNodes.push_back(parentNode->childNodes[childIdx]);
            backup_value(currentNode, -currentNode->get_value());
        }
        else if (description.isCollision) {
            // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
            collisionNodes.push_back(currentNode);
        }
        else {
            prepare_node_for_nn(currentNode, newNodes, inputPlanes);
        }
    }
}

void SearchThread::thread_iteration()
{
    create_mini_batch();
    if (newNodes.size() != 0) {
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
        backup_value(node, -node->get_value());
    }
    nodes.clear();
}

void prepare_node_for_nn(Node* newNode, vector<Node*>& newNodes, float* inputPlanes)
{
    // fill a new board in the input_planes vector
    // we shift the index by NB_VALUES_TOTAL each time
    board_to_planes(newNode->get_pos(), newNode->get_pos()->number_repetitions(), true, inputPlanes+newNodes.size()*NB_VALUES_TOTAL);

    // save a reference newly created list in the temporary list for node creation
    // it will later be updated with the evaluation of the NN
    newNodes.push_back(newNode);
}

void fill_nn_results(size_t batchIdx, bool is_policy_map, const SearchSettings* searchSettings, NDArray* valueOutputs, NDArray* probOutputs, Node *node)
{
    vector<Move> legalMoves = retrieve_legal_moves(node->get_child_nodes());
    assert(legalMoves.size() == node->get_number_child_nodes());
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
