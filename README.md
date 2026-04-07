README
Our work has built a communication-perception coupled collaborative perception module, which can effectively characterize the impact of the limited communication resources on the performance of collaborative perception. The platform takes the communication module as the main task and conducts joint communication-perception simulation with collaborative perception as a downstream task.

1. Briefly introduce our communication module 
   We design the link-level communication module in accordance with 3GPP TR 37.885. 
   The calculation of large-scale fading is in the  large_scale_fading.py  file, while small-scale fading is optional and integrated into the  calculate_throughput.py  file, which is used to calculate the throughput of time slots and subchannels. Among them, the small-scale fading refers to the design of the CDL channel of the Sionna open-source platform and makes simple modifications to adapt to the current platform. 
   The calculation results of the communication module will be passed into  optimize_comm.py  file. In this file, we propose code for optimizing cooperative sensing based on the binary search method, and users can also complete the joint optimization of communication and sensing by providing their own optimization methods. 

2. Collaborative Perception Task 
   The collaborative perception task is adapted to the open-source OpenCOOD platform, and only a small number of modifications are needed to integrate our communication module with it. Due to the limitations of the open-source license, we do not provide the corresponding code here, and users can search for and modify it themselves. 

Xinyue Wei, Sheng Gao, Kan Zheng, Jie Mei, Lu Hou
