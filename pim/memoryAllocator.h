#ifndef MEMORY_H
#define MEMORY_H
#include "constants.h"
namespace pim
{
  bool GPU[BUFFER_SIZE];
  bool Xbar[Numberofcrossbar*Numberofcolumns];

  int tracker_gpu_mem(int size)
  {
     int start_idx = 0;
     while(start_idx < BUFFER_SIZE-size+1)
     {
         bool found = false;
         int allocator_idx = 0;

         for(int allocator_idx = start_idx; allocator_idx < size; allocator_idx++)
         {
             if(GPU[allocator_idx] == true)
             {
                 found = true; break;
             }
         }
         start_idx = allocator_idx +1;
         if(found == false)
             return start_idx;
     }
     return -1;
           
  }

 
  RangeMask tracker_Xbar(int num_rows,int num_cols)
  {
     int num_rows_per_col = ceil((float)num_rows/(float)Numberofrows);
     int req_num_Xbar_cols = num_rows_per_col*num_cols;
     int start_idx = 0;
     RangeMask xbar_allotted;
  
     while(start_idx < Numberofcrossbar*Numberofcolumns - req_num_Xbar_cols + 1)
     {
         bool found = false;
         int allocator_idx = 0;

         for(allocator_idx = start_idx; allocator_idx < req_num_Xbar_cols; allocator_idx++)
         {
             if(Xbar[allocator_idx] == true)
             {
                 found = true; break;
             }
         }

         if(found == false){
             xbar_allotted.start.xbar_idx = start_idx / Numberofcolumns;
             xbar_allotted.start.col_idx = start_idx % Numberofcolumns;
             xbar_allotted.end.xbar_idx = (start_idx + req_num_Xbar_cols) / Numberofcolumns;
             xbar_allotted.end.col_idx = (start_idx + req_num_Xbar_cols) % Numberofcolumns;
             return xbar_allotted;
         }
         start_idx = allocator_idx +1;
     }

     xbar_allotted.start.xbar_idx = -1;
     xbar_allotted.start.col_idx = -1;
     xbar_allotted.end.xbar_idx = -1;
     xbar_allotted.end.col_idx = -1;
     return xbar_allotted;
           
  } 

}
#endif
