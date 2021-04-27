#ifndef FAIRYUTIL_H
#define FAIRYUTIL_H

#include <types.h>

using namespace std;


/**
 * @brief get_origin_square Returns the origin square for a valid ucciMove
 * @param uciMove uci-Move in string notation
 * @return origin square
 */
Square get_origin_square(const string &uciMove);

/**
 * @brief get_origin_square Returns the destination square for a valid ucciMove
 * @param uciMove uci-Move in string notation
 * @return destination square
 */
Square get_destination_square(const string &uciMove);

/**
 * @brief file_to_uci Returns the uci corresponding to the given file
 * @param file FILE to convert
 * @return uci corresponding to the file
 */
char file_to_uci(File file);

/**
 * @brief rank_to_uci Returns the uci corresponding to the given rank
 * @param rank Rank to convert
 * @return uci corresponding to the rank
 */
std::string rank_to_uci(Rank rank);

#endif //FAIRYUTIL_H
