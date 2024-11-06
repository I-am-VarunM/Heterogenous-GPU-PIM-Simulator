#ifndef GPUMAT_H
#define GPUMAT_H

#include<iostream>
#include "driver.cuh"
#include "constants.h"
#include "memoryAllocator.h"

namespace pim
{
   class GPUmat
   {
      int* data;
      int num_rows;
      int num_cols;
      int start_addr;
      
      //constructors
      //Only size as input
      public:
      GPUmat(int num_cols,int num_rows = 1): num_rows(num_rows), num_cols(num_cols)
      {
          start_addr = tracker_gpu_mem(num_cols*num_rows);
          data = (int*)malloc(num_rows*num_cols);
        
          load_values_GPU(data, start_addr, num_rows*num_cols);
      }

      //Take a 2D standard vector as input
      GPUmat(std::vector<std::vector<int>> data_in)
      {
         
        num_rows = data_in.size();
        num_cols = num_rows < 1?0:data_in[0].size();
        start_addr = tracker_gpu_mem(num_cols*num_rows);
        data = (int*)malloc(num_rows*num_cols);
        for(int idx = 0; idx < num_rows*num_cols;idx++)
            data[idx] = data_in[idx/num_cols][idx%num_cols];
        load_values_GPU(data, start_addr, num_rows*num_cols);
      }
      
       
      //Utility functions
      int* begin()
      {
         return data;
      }  
      int* end()
      {
         return data + num_rows*num_cols;
      }    
      int size()
      {
         return num_rows*num_cols;
      }

      int address()
      {
        return start_addr;
      }

      //Operation overloading to access random position

      //Class to Access element of a vector at position x
      // Necessary to support weight[][]
      //Add code later to support mat1[x][y] = 10
      class ElementAccessor
      {
          friend class GPUmat;
          GPUmat& vec;
          int pos_row;
          int pos_col;
          
          //Constructor using weight 
          ElementAccessor(GPUmat vec, int pos_row): vec(vec), pos_row(pos_row) {}
          
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
     GPUmat operator*(GPUmat vec_in)
     {
         GPUmat res = GPUmat(num_rows,num_cols); 
         GPUmatmult(start_addr,vec_in.address(),res.address(),num_cols,num_rows);
         return res;
     }

     //Scalar Division
    GPUmat softmax()
    {
        GPUmat res = GPUmat(num_rows,num_cols);
        softmaxDr(start_addr, res.address(),num_rows,num_cols);
        return res;
    }

    GPUmat operator/(int scale)
    {
        GPUmat res = GPUmat(num_rows,num_cols);
        scalardivDr(start_addr,res.address(),num_rows, num_cols,scale);
        return res;
    }

    GPUmat transpose()
    {
        std_mat transposed_data(num_cols);
        for(int col_idx = 0; col_idx < num_cols; col_idx ++)
        {
            for(int row_idx = 0; row_idx < num_rows; row_idx++)
            {
                transposed_data[col_idx].push_back(data[row_idx*num_cols + col_idx]);
            }
        }
        GPUmat res = GPUmat(transposed_data);
        return res;
    }

};

     

  }

#endif
