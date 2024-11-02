#include "weight.h"
#include "GPUmat.h"
typedef std::vector<std::vector<int>> std_mat;
std_mat multi_head_attention(std_mat input, std_mat Q,std_mat V,std_mat K,int d_model)
{
    //Parallel execution of steps
    //Converts std::vector<std::vector<>> to weight data type
    pim::weight Query_w(Q);
    pim::weight Key_w(K);
    pim::weight Value_w(V);

    //Obtains Q,K,V
    pim::GPUmat Query = input*Query_w; 
    pim::GPUmat Key = input*Key_w;
    pim::GPUmat Value = input*Value_w;

    //SOlve until softmax 
    pim::GPUmat QK_T = Query*Key.transpose();
    pim::GPUmat QK_T_scaled = QK_T/d_model;
    pim::GPUmat softmax_QK_T = QK_T_scaled.softmax();
    pim::GPUmat output = softmax_QK_T*Value;

    return output;
}
std_mat feed_forward()
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
    
}

