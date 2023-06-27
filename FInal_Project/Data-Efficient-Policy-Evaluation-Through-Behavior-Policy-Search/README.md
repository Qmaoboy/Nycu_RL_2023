# Data-Efficient-Policy-Evaluation-Through-Behavior-Policy-Search
## Replication of Paper 'Data-Efficient Policy Evaluation Through Behavior Policy Search'
In this paper the original BPG algo as below:  
![algo](https://github.com/Qmaoboy/Data-Efficient-Policy-Evaluation-Through-Behavior-Policy-Search/assets/101452682/426a8085-8f73-41f9-8b00-a7a961a28728)
## Experiment Setting
In experiment setting , we apply BPG on two different Tasks of OpenAI Gym Cartpole and Acrobot.
we apply the algorithm as below: 
![image](https://github.com/Qmaoboy/Data-Efficient-Policy-Evaluation-Through-Behavior-Policy-Search/assets/101452682/409bf861-06e1-4c3a-bd5b-b93ce1ecc4da)

So in experiment setting we apply two setting for BPG and off-policy-evaluation respectively.  
$Step_b$ = 64 : BPG  
$Step_b$ = 0  : Off-policy REINFORCE  
Finally, BPG will converge to a behavior policy that locally minimizes the variance and ideally converges to the globally optimal behavior policy within the parametrization of $\pi_e$  
## Task 1: Cartpole  
  random seed : 10  
  Learning Rate lr : 3e-5  
  Discount factor γ : 0.95  
  optimizer : Adam (lr)  
  Scheduler : StepLR (Step size=100 ,γ = 0.9)  
  Behavior update step n : 64  
  
Experiment result:  
![353821902_637236071658815_7902503122007230345_n](https://github.com/Qmaoboy/Data-Efficient-Policy-Evaluation-Through-Behavior-Policy-Search/assets/101452682/8756b204-94e3-44d2-8167-e01cf766890e)
Result GIF:
![Cartpole_gif](gif/Carpole.gif)
## Task 2: Acrobot  
  random seed : 10  
  Learning Rate lr :3e-5  
  Discount factor γ : 0.95  
  optimizer : Adam (lr)  
  Scheduler : StepLR (Step size=100 ,γ = 0.9)  
  Behavior update step n : 64  
Experiment result:  
![acrobot_result](https://github.com/Qmaoboy/Data-Efficient-Policy-Evaluation-Through-Behavior-Policy-Search/assets/101452682/725b83be-0849-42fb-87b9-8481c91ec618)
Result GIF:
![Cartpole_gif](gif/Acrobot.gif)


