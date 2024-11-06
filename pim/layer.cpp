#include<vector>
typedef std::vector<std::vector<int>> std_mat;
#include "weight.h"
#include "GPUmat.h"
pim::GPUmat multi_head_attention(std_mat input, std_mat Q,std_mat V,std_mat K,int d_model)
{
    //Parallel execution of steps
    printf(" In function multi head\n");
    //Converts std::vector<std::vector<>> to weight data type
    pim::weight Query_w(Q);
    pim::weight Key_w(K);
    pim::weight Value_w(V);

    
    printf(" Initialised weights\n");
    //Obtains Q,K,V
    pim::GPUmat Query = Query_w*input; 
    pim::GPUmat Key = Key_w*input;
    pim::GPUmat Value = Value_w*input;

    printf(" Obtained Query,key,value\n");
    //SOlve until softmax 
    pim::GPUmat QK_T = Query*Key.transpose();
    pim::GPUmat QK_T_scaled = QK_T/d_model;
    pim::GPUmat softmax_QK_T = QK_T_scaled.softmax();
    pim::GPUmat output = softmax_QK_T*Value;
    printf("Obtained output\n");
    return output;
    //return Value;
}
/*std_mat feed_forward()
{

}
std_mat convolutional()
{

}
std_mat maxpool()
{

}
std_mat activation()
{

}
std_mat normalisation()
{
    
}*/

int main()
{
   std_mat input = {{1,1,1,1},{1,1,1,1},{1,1,1,1}};
   std_mat Q = {{1,1,1,1},{1,1,1,1},{1,1,1,1}};
   std_mat K = {{1,1,1,1},{1,1,1,1},{1,1,1,1}};
   std_mat V = {{1,1,1,1},{1,1,1,1},{1,1,1,1}};
   int d_model = 4;
   multi_head_attention(input,Q,V,K,d_model);

   return 0;
}
