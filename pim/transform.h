#include "weightMat.h"
#include "matrix.h"

namespace pim
{
	 template <class T>;
	 pim::matrix<T> encode(pim::matrix<T> &X,pim::weight<T> &Wq,pim::weight<T> &Wk,pim::weight<T> &Wv,T dk)
	 {
	        pim::matrix<T> Q = X*Wq;//make it all parallel: how to do it on gpu and how to indicate the steps
	        pim::matrix<T> K = X*Wk;
		pim::matrix<T> V = X*Wv;//mat*weightmat
		pim::matrix<T> KT = K.transpose();
		pim::matrix<T> QKT = Q*KT;//mat*mat
		pim::matrix<T> S = QKT.softmax();//softmax as a matrix function
		pim::matrix<T> out = S*V.transpose();	
	 	
		return out;
	 }


}
