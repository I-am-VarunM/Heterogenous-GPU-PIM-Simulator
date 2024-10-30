namespace pim
{
  bool GPU[BUFFER_SIZE];
  bool Xbar[Numberofcrossbars*Numberofcolumns];

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
         if(found == true)
             return start_idx;
     }
     return -1;
           
  }

 
  RangeMask tracker_Xbar(int size)
  {
     int req_num_Xbar = size/Numberofcolumns;
     int start_idx = 0;
     RangeMask xbar_allotted;
  
     while(start_idx < req_num_Xbar-size+1)
     {
         bool found = false;
         int allocator_idx = 0;

         for(int allocator_idx = start_idx; allocator_idx < req_num_Xbar; allocator_idx++)
         {
             if(Xbar[allocator_idx] == true)
             {
                 found = true; break;
             }
         }
         start_idx = allocator_idx +1;

         if(found == true){
             xbar_allotted.start.num_xbar = start_idx / Numberofcolumns;
             xbar_allotted.start.num_col = start_idx % Numberofcolumns;
             xbar_allotted.end.num_xbar = (start_idx + req_num_Xbar) / Numberofcolumns;
             xbar_allotted.end.num_col = (start_idx + req_num_Xbar) % Numberofcolumns;
             return xbar_allotted;
         }
     }

     xbar_allotted.start.num_xbar = -1;
     xbar_allotted.start.num_col = -1;
     xbar_allotted.end.num_xbar = -1;
     xbar_allotted.end.num_col = -1;
     return xbar_allotted;
           
  } 

}
