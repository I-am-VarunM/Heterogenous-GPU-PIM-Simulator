#include<iostream>
#include "driver.cu"
#include "constants.h"
namespace pim
{
   class GPUmat
   {
      std::vector<int> data;
      int num_rows;
      int num_cols;
      int start_addr;
      
      //constructors
      //Only size as input
      GPUmat(int num_cols,int num_rows = 1): num_rows(num_rows), num_cols(num_cols)
      {
          start_addr = tracker_gpu_mem(num_cols*num_rows);
          data.resize(num_rows*num_cols);
        
          load_values_GPU(data, start_addr, num_rows*num_cols);
      }

      //Take a 2D standard vector as input
      GPUmat(std::vector<std::vector<int>> data_in)
      {
        for(const auto& row:data_in)
           data.insert(data.end(),row.begin(),row.end());
        
        num_rows = data_in.size();
        num_cols = num_rows >= 1?0:data_in[0].size();
        start_addr = tracker_gpu_mem(num_cols*num_rows);

        load_values_GPU(data, start_addr, num_rows*num_cols);
      }
    
      //Utility functions
      int* begin()
      {
         return data.begin();
      }  
      int* end()
      {
         return data.end();
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

          weight vec;
          int pos_row;
          int pos_col;
          
          //Constructor using weight 
          ElementAccessor(weight vec, int pos_row): vec(vec), pos_row(pos_row) {}
          
          //Copy constructor
          ElementAccessor(ElementAccessor& parentAccessor)
          {
              vec = parentAccessor.vec;
              pos_row = parentAccessor.pos_row;
              pos_col = parentAccessor.pos_col;
          }

          //Operation overloading for []
          int operator[](int index)
          {
              pos_col = index;
              return vec.data[pos_row*num_cols + pos_col];
          }
         
          //Operation overloading for mat[x][y] = 2
          ElementAccessor* operator[](int index)
          {
             pos_col = index;
             return ElementAccessor(*this);
          } 

          void operator=(int value)
          {
              vec.data[pos_row*num_cols + pos_col] = value;
          }          

          void operator=(ElementAccessor* value)
          {
              int value_pos_row = value.pos_row;
              int value_num_cols = value.vec.num_cols;
              int value_num_rows = value.vec.num_rows;
              vec.data[pos_row*num_cols + pos_col] = value.data[value_pos_row*value_num_cols + value_num_rows];
          }

      }

      ElementAccessor* operator[](int index) {
        if (index >= length) {
            throw std::out_of_range("Index out of range");
        }
        return ElementAccessor(*this,index);
      }

     //Operation overloading for *
     GPUmat operator*(GPUmat vec_in)
     {
         res = GPUmat(num_rows,num_cols); 
         GPUmatmult(*this.start_addr,vec_in.address(),res.address(),*this.num_cols,*this.num_rows);
         return res;
     }

     //Scalar Division
    GPUmat softmax()
    {
        res = GPUmat(num_rows,num_cols);
        softmaxDr(start_addr, res.address(),num_rows,num_cols);
    }

    GPUmat operator/(int scale)
    {
        res = GPUmat(num_rows,num_cols);
        scalardivDr(start_addr,res.address(),num_rows, num_cols,scale);
    }

    GPUmat transpose()
    {
        std_mat transposed_data(num_cols);
        for(int col_idx = 0; col_idx < num_cols; col_idx ++)
        {
            for(int row_idx = 0; row_idx < num_rows; row_idx++)
            {
                transposed_data[col_idx].push_back(data[row_idx][col_idx])
            }
        }
        res = GPUmat(transposed_data);
    }
}

     

  }


