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
 * @file: mctsagent.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "mctsagent.h"
#include "../evalinfo.h"
#include "movegen.h"
#include "inputrepresentation.h"
#include "outputrepresentation.h"
#include "constants.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

void MCTSAgent::run_single_playout() //Board &pos) //, int i) //Node *rootNode)
{
    std::cout << "hello :) " << std::endl;
}

void MCTSAgent::expand_root_node_multiple_moves(const Board &pos, const std::vector<Move> &legalMoves)
{
//    bool isLeaf = false;  // initialize is_leaf by default to false
//    // [value, policy_vec] = self.nets[0].predict_single(state.get_state_planes())  # start a brand new tree
//    float value = 0.42f;

//    Eigen::VectorXf policyVec = Eigen::VectorXf::Random(NB_LABELS);

//    // extract a sparse policy vector with normalized probabilities
//    Eigen::VectorXf policyVecSmall = get_probs_of_move_list(policyVec, legalMoves, pos.side_to_move());

//    /*
//    chess_board = state.get_pythonchess_board()
//    if self.enhance_captures:
//        self._enhance_captures(chess_board, legalMoves, p_vec_small)

//    if self.enhance_checks:
//        self._enhance_checks(chess_board, legalMoves, p_vec_small)

//    # create a new root node
//    self.root_node = Node(chess_board, value, p_vec_small, legalMoves, is_leaf, clip_low_visit=False)
//    */

//    this->rootNode = new Node(value, pos, policyVecSmall, legalMoves, isLeaf);
//    policyVec.maxCoeff();
}

void MCTSAgent::select_node_to_extend()
{

}

void MCTSAgent::select_node(Node &parentNode)
{
    /*
    def _select_node(self, parent_node: Node):
        """
        Selects the best child node from a given parent node based on the q and u value
        :param parent_node:
        :return: node - Reference to the node object which has been selected
                        If this node hasn't been expanded yet, None will be returned
                move - The move which leads to the selected child node from the given parent node on forward
                node_idx - Integer idx value indicating the index for the selected child of the parent node
        """

        if parent_node.mate_child_idx:  # check first if there's an immediate mate in one move possible
            child_idx = parent_node.mate_child_idx
        else:
            # find the move according to the q- and u-values for each move
            if not self.use_oscillating_cpuct:
                pb_c_base = 19652
                pb_c_init = self.cpuct / 2 # self.cpuct

                cpuct = math.log((parent_node.n_sum + pb_c_base + 1) / pb_c_base) + pb_c_init
            else:
                cpuct = self.cpuct
            # calculate the current u values
            # it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
            u_value = (
                cpuct * parent_node.policy_prob * (np.sqrt(parent_node.n_sum) / (1 + parent_node.child_number_visits))
            )

            # if parent_node.n_sum % 2 == 0:
            #     # get 2nd best
            # distribution = (parent_node.q_value + u_value)
            # distribution /= distribution.sum()
            # # try:
            # child_idx = np.random.choice(range(parent_node.nb_direct_child_nodes), p=distribution) #distribution.argmax()
            #     # except Exception:
            #     #     raise Exception(self.use_pruning, parent_node.q_value)
            #     # distribution[child_idx] = 0
            #     # child_idx = distribution.argmax()
            #     # child_idx = np.random.randint(parent_node.nb_direct_child_nodes)
            # else:
            child_idx = (parent_node.q_value + u_value).argmax()

        return parent_node.child_nodes[child_idx], parent_node.legal_moves[child_idx], child_idx
        */
}

MCTSAgent::MCTSAgent(NeuralNetAPI *net, SearchSettings searchSettings, SearchLimits searchLimits, PlaySettings playSettings)
{
    this->net = net;
    this->searchSettings = searchSettings;
    this->searchLimits = searchLimits;
    this->playSettings = playSettings;
}

EvalInfo MCTSAgent::evalute_board_state(const Board &pos)
{
    std::vector<Move> legalMoves;
    for (const ExtMove& move : MoveList<LEGAL>(pos)) {
        legalMoves.push_back(move);
    }

    if (legalMoves.size() > 1) {
        expand_root_node_multiple_moves(pos, legalMoves);
    }

    run_mcts_search(pos);

    float input_planes[34][8][8];
    float *input_planes_start = &input_planes[0][0][0];
    std::fill(input_planes_start, input_planes_start+34*8*8, 0.0f);

    board_to_planes(pos, 0, true, input_planes_start);

    float sum = 0;
    for (int i = 0; i < 34*8*8; ++i) {
//    std::cout << "input_planes" << *(input_planes_start+i) << std::endl;
    sum += *(input_planes_start+i);
    }
    /*
//    std::cout << "sum" << sum << std::endl;
    Symbol net;
//    const std::string prefix = "/home/queensgambit/Programming/Deep_Learning/models/risev2/";
    const std::string prefix = "/home/queensgambit/Programming/Deep_Learning/models/orig_resnet_4_value_8_policy/"; //home/queensgambit/Programming/Deep_Learning/CrazyAra_Fish/";
//    const std::string model_json_file = prefix + "symbol/model-1.19246-0.603-symbol.json"; //model-1.19246-0.603-symbol.json";
    const std::string model_json_file = prefix + "symbol/model-1.21119-0.600-symbol.json"; //model-1.19246-0.603-symbol.json";
    LG << "Loading the model from " << model_json_file << std::endl;
    net = Symbol::Load(model_json_file);

    Context global_ctx = Context::cpu();
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::vector<std::string> output_labels;

//    const std::string model_parameters_file = prefix + "params/model-1.19246-0.603-0223.params"; //model-1.19246-0.603-0223.params";
    const std::string model_parameters_file = prefix + "params/model-1.21119-0.600-0222.params"; //model-1.19246-0.603-0223.params";

    LG << "Loading the model parameters from " << model_parameters_file << std::endl;
    std::map<std::string, NDArray> parameters;
    NDArray::Load(model_parameters_file, 0, &parameters);
    for (const auto &k : parameters) {
      if (k.first.substr(0, 4) == "aux:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        aux_map[name] = k.second.Copy(global_ctx);
      }
      if (k.first.substr(0, 4) == "arg:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        args_map[name] = k.second.Copy(global_ctx);
      }
    }
    // WaitAll is need when we copy data between GPU and the main memory
    NDArray::WaitAll();

    Executor *executor;
    std::vector<index_t> input_dims;
    input_dims.push_back(1);
    input_dims.push_back(34);
    input_dims.push_back(8);
    input_dims.push_back(8);

    Shape input_shape(input_dims);
    // Create an executor after binding the model to input parameters.
    args_map["data"] = NDArray(input_shape, global_ctx, false);
    executor = net.SimpleBind(global_ctx, args_map, std::map<std::string, NDArray>(),
  std::map<std::string, OpReqType>(), aux_map);
    LG << ">>>> Bind successfull! >>>>>>";

//    NDArray image_data = NDArray(input_shape, global_ctx, false);
    std::vector<mx_float> zeroes(34*8*8);

//    image_data.SyncCopyFromCPU(zeroes, input_shape.Size());


    // populates v vector data in a matrix of 1 row and 4 columns
     mxnet::cpp::NDArray image_data {input_planes_start, input_shape, global_ctx};

    std::cout << "image data" << image_data << std::endl;
    image_data.CopyTo(&(executor->arg_dict()["data"]));

    // Run the forward pass.
    executor->Forward(false);

    // The output is available in executor->outputs.
    auto array = executor->outputs[1].Copy(Context::cpu());


     Find out the maximum accuracy and the index associated with that accuracy.
     This is done by using the argmax operator on NDArray.
    auto predicted = array.ArgmaxChannel();

    predicted.WaitToRead();

    int best_idx = predicted.At(0, 0);
    */
    float value;
    Eigen::VectorXf prob_vec;

    int best_idx = 0; //net->predict_single(input_planes_start, value, prob_vec);
//    best_accuracy = array.At(0, best_idx);

//    std::cout << "array " << array << std::endl;
    LABELS_MIRRORED[77] ="%§";
    Constants::init();
    if (pos.side_to_move() == WHITE) {
        std::cout << "predicted " << best_idx << " " <<  LABELS[best_idx] << std::endl;
    }
    else {
        std::cout << "predicted " << best_idx << " move" <<  LABELS_MIRRORED[best_idx] << std::endl;
    }
    EvalInfo eval_info;
    eval_info.centipawns = value_to_centipawn(this->rootNode->getValue());
    eval_info.depth = 42;
    eval_info.legalMoves = this->rootNode->getLegalMoves();
//    eval_info.policyProbSmall = this->rootNode->getPVecSmall();

    return eval_info;
}

void MCTSAgent::run_mcts_search(const Board &pos)
{
    const int num_threads = 32;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; ++i) {
//        go();
        threads[i] = std::thread(run_single_playout); //, pos); //, 3); //this->rootNode);
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    
}





