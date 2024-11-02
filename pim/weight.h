#include<iostream>
#include "driver.cu"
#include "constants.h"
typedef std::vector<std::vector<int>> std_mat;
namespace pim
{
   class weight 
   {
      std::vector<int> data;
      int num_rows;
      int num_cols;
      RangeMask XbRange;
      
      //constructors
      //Only size as input
      weight(int num_cols,int num_rows = 1): num_rows(num_rows), num_cols(num_cols)
      {
          XbRange = tracker_Xbar(num_rows,num_cols);
          data.resize(num_rows*num_cols,0);
          loadweightstopim(data, xbar_range,num_rows,num_cols);
      }

      //Take a 2D standard vector as input
      weight(std::vector<std::vector<int>> data_in)
      {
        xbar_range = tracker_Xbar(num_rows,num_cols);
        for(const auto& row:data_in)
           data.insert(data.end(),row.begin(),row.end());
        
        num_rows = data_in.size();
        num_cols = num_rows >= 1?0:data_in[0].size();
        
        loadweightstopim(data, xbar_range,num_rows,num_cols);
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
     GPUmat operator*(std::vector<std::vector<int>> vec_in)
     {
         int* vec_in_ptr = (int*) malloc(vec_in.size()*vec_in[0]*size());
         for(int idx = 0; idx < vec_in.size(); idx++)
             vec_in_ptr[idx] = vec_in[idx/vec_in.size()][idx % vec_in.size()];
         GPUmat vec_out(num_cols,num_rows);
         matrix_multiplication(vec_in_ptr, num_cols, num_rows,XbRange.Xstart,XbRange.Xend,'G',vec_out.address());
         return vec_out;
     }


  }

}
