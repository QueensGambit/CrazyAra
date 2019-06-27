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
 * @file: searchthread.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "searchthread.h"
#include "inputrepresentation.h"
#include "outputrepresentation.h"
#include "uci.h"

SearchThread::SearchThread(NeuralNetAPI *netBatch, unsigned int batchSize, const float virtualLoss, unordered_map<Key, Node *> *hashTable):
    netBatch(netBatch), batchSize(batchSize), virtualLoss(virtualLoss), isRunning(false), hashTable(hashTable)
{
    // allocate memory for all predictions and results
    inputPlanes = new float[batchSize * NB_VALUES_TOTAL];
    NDArray valueOutput = NDArray(Shape(batchSize, 1), Context::cpu());
    NDArray probOutputs = NDArray(Shape(batchSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH), Context::cpu());
//    states = nullptr;
//    states = StateListPtr(new std::deque<StateInfo>(1)); // Drop old and create a new one
}

void SearchThread::go()
{
    isRunning = true;
//    cout << "rootNode" << endl;
//    cout << rootNode << endl;

//    while(isRunning) {
    for (int i = 0; i < 128; ++i) {
        create_mini_batch();
//        cout << "predict" << endl;k
        netBatch->predict(inputPlanes, valueOutputs, probOutputs);
//        cout << "set NN result to childs" << endl;
        set_NN_results_to_child_nodes();
//        cout << "backup values" << endl;
        backup_value_outputs(virtualLoss);
        backup_collisions(virtualLoss);
        rootNode->numberVisits = sum(rootNode->childNumberVisits);
//        isRunning = false;
//        if (rootNode->numberVisits % 100) {
//            cout << rootNode->numberVisits << endl;
//        }
//        if (rootNode->numberVisits > 400) {
//            break;
//        }
    }
//    cout << "rootNode" << endl;
//    cout << rootNode << endl;
}

void SearchThread::run_single_playout()
{
    //    create_mini_batch();
}

void SearchThread::setRootNode(Node *value)
{
    rootNode = value;
}

inline Node* SearchThread::get_new_child_to_evaluate(unsigned int &childIdx, bool &isCollision, bool &isTerminal, size_t &depth)
{
    Node *currentNode = rootNode;
    Node *nextNode;

    // traverse the tree until you get to a new unexplored node
//    currentNode = rootNode; //rootNode;

    depth = 0;

    while (true) {
//        cout << currentNode->pos.fen() << endl;
//        cout << currentNode->pos << endl;
        childIdx = currentNode->select_child_node(2.5);
        currentNode->apply_virtual_loss_to_child(childIdx, virtualLoss);
        nextNode = currentNode->get_child_node(childIdx);
        depth++;
        if (nextNode == nullptr) {
            isCollision = false;
            isTerminal = false;
            return currentNode;
        }
        if (nextNode->isTerminal) {
            isCollision = false;
            isTerminal = true;
            return currentNode;
        }
        if (!nextNode->hasNNResults) {
            isCollision = true;
            isTerminal = false;
            return currentNode;
        }
        currentNode = currentNode->childNodes[childIdx];
    }

}

void SearchThread::set_NN_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: newNodes) {
        if (!node->isTerminal) {
            get_probs_of_move_list(batchIdx, probOutputs, node->legalMoves, node->pos.side_to_move(), true, node->policyProbSmall);
            node->value = valueOutputs.At(batchIdx, 0);
//            cout << "node->value: " << node->value << endl;
            node->hasNNResults = true;
        }
//        node->parentNode->waitForNNResults[node->childIdxOfParent] = 0;
//        node->parentNode->numberWaitingChildNodes--;
        ++batchIdx;
        hashTable->insert({node->pos.key(), node});
    }
}

void SearchThread::backup_value_outputs(const float virtualLoss)
{
//    size_t batchIdx = 0;
    for (auto node: newNodes) {
        node->parentNode->backup_value(node->childIdxOfParent, virtualLoss, -node->value);
//        ++batchIdx;
    }
    newNodes.clear();

//    batchIdx = 0;
    for (auto node: terminalNodes) {
        node->parentNode->backup_value(node->childIdxOfParent, virtualLoss, -node->value);
//        ++batchIdx;
    }
    terminalNodes.clear();

}

void SearchThread::backup_collisions(const float virtualLoss)
{
//    size_t batchIdx = 0;
    for (auto node: collisionNodes) {
//        node->parentNode->backup_collision(node->childIdxOfParent, virtualLoss);
//        ++batchIdx;
    }
    collisionNodes.clear();
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to the mini-batchel
    Node *parentNode;
    unsigned int childIdx;
    bool isCollision;
    bool isTerminal;

//    cout << "batchSize " << batchSize << endl;
    size_t depth;
    size_t numberNewNodes = 0;

    for (size_t i = 0; i < batchSize; ++i) {
//        cout << "getNewChildToEval" << endl;
        parentNode = get_new_child_to_evaluate(childIdx, isCollision, isTerminal, depth);
//        cout << "move " << UCI::move(parentNode->legalMoves[childIdx], false) << " depth " << depth << endl;
//        cout << "parentNode->numberWaitingChildNodes" << parentNode->numberWaitingChildNodes << endl;
//        cout << "parentNode->nbDirectChildNodes" << parentNode->nbDirectChildNodes << endl;

        if(isTerminal) {
            terminalNodes.push_back(parentNode->childNodes[childIdx]);
//            cout << ">>>>>>>>>>>>isTerminal!!!!!" << endl;
        }
        else if (!isCollision) {
//        if (parentNode->numberWaitingChildNodes < parentNode->nbDirectChildNodes) {
//            parentNode->waitForNNResults[childIdx] = -INFINITY;
//            parentNode->numberWaitingChildNodes++;

            StateInfo* newState = new StateInfo;
            Board newPos(parentNode->pos);
            newPos.do_move(parentNode->legalMoves[childIdx], *newState);

    //        currentNode->childNodes.push_back(Node());
            Node *newNode = new Node(newPos, parentNode, childIdx);

            // save a reference newly created list in the temporary list for node creation
            // it will later be updated with the evaluation of the NN
            newNodes.push_back(newNode);

            // connect the Node to the parent
            parentNode->childNodes[childIdx] = newNode;

            // fill a new board in the input_planes vector
            // we shift the index by NB_VALUES_TOTAL each time
            board_to_planes(newPos, 0, true, inputPlanes+numberNewNodes*NB_VALUES_TOTAL); //input_planes_start);
            ++numberNewNodes;
        }
        else {
//            cout << " node collision " << i <<  endl;
            // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
            collisionNodes.push_back(parentNode->childNodes[childIdx]);
//            break;
        }
    }

//    if (collisionNodes.size() > 0) {
//        cout << "collision" << collisionNodes.size() << endl;
//    }
//    cout << "created mini-batch" << endl;
//    currentNode->[childIdx] = Node();
}

