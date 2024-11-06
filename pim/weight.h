#ifndef WEIGHT_H
#define WEIGHT_H

#include<iostream>
#include "GPUmat.h"
#include "memoryAllocator.h"

typedef std::vector<std::vector<int>> std_mat;
namespace pim
{
   class weight 
   {
      int* data;
      int num_rows;
      int num_cols;
      RangeMask xbar_range;
      
      //constructors
      //Only size as input
      public:
      weight(int num_cols,int num_rows = 1): num_rows(num_rows), num_cols(num_cols)
      {
          xbar_range = tracker_Xbar(num_rows,num_cols);
          data = (int*) calloc(num_rows*num_cols,sizeof(int));
          loadweightstopim(data, xbar_range,num_rows,num_cols);
      }

      //Take a 2D standard vector as input
      weight(std::vector<std::vector<int>> data_in)
      {
        num_rows = data_in.size();
        num_cols = num_rows < 1?0:data_in[0].size();
        xbar_range = tracker_Xbar(num_rows,num_cols);
        data = (int*) malloc(num_rows*num_cols*sizeof(int));
        for(int idx = 0; idx < num_rows*num_cols; idx++){
            data[idx] = data_in[idx/num_cols][idx%num_cols];
        }
        loadweightstopim(data, xbar_range,num_rows,num_cols);
      }
    
      
      //Utility functions
      int* begin()
      {
         return data;
      }  
      int* end()
      {
         return data+num_rows*num_cols;
      }    
      int size()
      {
         return num_rows*num_cols;
      }
      //Operation overloading to access random position

      //Class to Access element of a vector at position x
      // Necessary to support weight[][]
      //Add code later to support mat1[x][y] = 10
      class ElementAccessor
      {

          friend class weight;
          weight& vec;
          int pos_row;
          int pos_col;
         
          //Constructor using weight 
          ElementAccessor(weight& vec, int pos_row): vec(vec), pos_row(pos_row) {}
          
          public:
          //Copy constructor
          ElementAccessor(ElementAccessor& parentAccessor)
             :vec(parentAccessor.vec),
              pos_row(parentAccessor.pos_row),
              pos_col(parentAccessor.pos_col){
          }

          //Operation overloading for []
          int operator[](int index) const
          {
              return vec.data[pos_row*vec.num_cols + index];
          }
         
          //Operation overloading for mat[x][y] = 2
          ElementAccessor operator[](int index) 
          {
             pos_col = index;
             return ElementAccessor(*this);
          } 

          void operator=(int value)
          {
              vec.data[pos_row*vec.num_cols + pos_col] = value;
          }          

          void operator=(ElementAccessor* value)
          {
              int value_pos_row = value->pos_row;
              int value_num_cols = value->vec.num_cols;
              int value_num_rows = value->vec.num_rows;
              vec.data[pos_row*vec.num_cols + pos_col] = value->vec.data[value_pos_row*value_num_cols + value_num_rows];
          }

      };

      ElementAccessor operator[](int index) {
        if (index >= num_rows) {
            throw std::out_of_range("Index out of range");
        }
        return ElementAccessor(*this,index);
      }

     //Operation overloading for *
     GPUmat operator*(std::vector<std::vector<int>> vec_in)
     {
         int* vec_in_ptr = (int*) malloc(vec_in.size()*vec_in[0].size()*sizeof(int));
         for(int idx = 0; idx < vec_in.size(); idx++)
             vec_in_ptr[idx] = vec_in[idx/vec_in.size()][idx % vec_in.size()];
         GPUmat vec_out(num_cols,num_rows);
         matrix_multiplication(vec_in_ptr, num_cols, num_rows,xbar_range.start.xbar_idx,xbar_range.end.xbar_idx,'G',vec_out.address());
         return vec_out;
     }


  };

}
#endif
