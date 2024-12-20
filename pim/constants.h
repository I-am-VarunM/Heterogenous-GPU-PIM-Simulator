#ifndef CUDAPIM_CONSTANTS_H
#define CUDAPIM_CONSTANTS_H

#include <cstdint>
#include <vector>
typedef std::vector<std::vector<int>> std_mat;

namespace pim {


    /**Embeddings length**/
    constexpr int EMBEDDINGS_SIZE = 512;
    constexpr size_t NUM_BITS = 2;
    constexpr size_t DECIMAL_SIZE = 1;
    
    const int Numberofcolumns = 256;
    const int Numberofrows = 256;
    const int Numberofcrossbar = 20;

    const int BUFFER_SIZE = Numberofcolumns*Numberofrows*Numberofcrossbar*20;
     /* Represents a generic range-based mask (e.g., {start, start + step, ..., stop}, inclusive)
     */
    struct RangeMask {
          struct xbar{
              int xbar_idx;
              int col_idx;

          }start,end;
    };

    /** A mask for all rows */
    #define ALL_ROWS RangeMask(0, CROSSBAR_HEIGHT - 1, 1)
    /** A mask for all crossbars */
    #define ALL_CROSSBARS RangeMask(0, NUM_CROSSBARS - 1, 1)

    /**
     * The different types of micro-operations
     */
    enum MicrooperationType{
        READ, WRITE, LOGIC, MASK
    };

    /**
     * Represents a vector address in the Xbar memory. A nullptr is represented with reg = -1.
     */
    struct addressXbar {

        /** The interval [startArray, endArray] of arrays containing the address */
        size_t startArray, endArray;


    };



}

#endif
