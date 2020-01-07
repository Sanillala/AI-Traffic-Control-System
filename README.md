# AI-Traffic-Control-System
This project was completed in partial fulfilment of my engineering degree.

## **Outline:**

The system utilises SUMO traffic simulator and Python 3 with TensorFlow. The system is developed for a minor arterial road intersection 
with traffic lights with turning signals. Utilising the reinforcement learning algorithm called Deep Q-learning, the system tries to choose the optimal traffic duration and phase to minimise traffic waiting 
times around the intersection. 
A 7 second yellow interval was employed, and green durations were adjusted between 10 s, 15 s and 20 s, depending on the vehicle demand. 
This system is a modified and adapted system developed by [1] as well as extracts from work done by [2, 3]. 
A more realistic approach was undertaken when developing this system with regards to real world sensors and data. 
Induction loop sensors were considered and thus the data from these sensors is what is used to drive the system. 

## **The Agent:**

**Framework:** Deep Q-Learning

**Environment:** A 4-way intersection with 2 incoming lanes and 2 outgoing lanes per arm. The traffic signals include a turning signal. 
500 cars were used for the simulation

**Sensors:** 3 induction loop sensors were placed 21 m apart per approaching lane to the intersection. Totalling 12 sensors used for 
the intersection. These sensors are used to obtain the waiting times, vehicle occupancy, vehicle position and vehicle velocity 
used by the agent to obtain the reward and develop the state. 

**State:** The state is made up of the position and velocity data received from the induction loop sensors as well as the 
current traffic phase and current duration. 

**Action:** The actions are the traffic phases. Since a turning signal is included, there are 4 possible phases for the intersection. 
However, the phase durations were as consider. 10 s, 15 s and 20 s green phase durations were used in this system. 
Therefore, the total number of actions were 12. 

**Training:** Q-learning was used with an Adaptive Momemnt Estimation (ADAM) optimiser. Additionally, Experience Replay techniques were also used so that the system 
could learn over time based on its previous memory. 

**Reward:** The reward is based on the vehicle occupancy on the induction loop sensors. These sensors given the time the 
vehicle was on the sensor thus, this is used to determine the reward. The waiting time of the vehicles on each sensor is 
summed to obtain the total waiting time. The goal of the system was to decrease the total waiting time as much as possible. 

**Action Policy:** Since the system is developed to learn over time, an epsilon greedy action possible is applied. 


## **Requirements to run the code:**
•	Python 

•	TensorFlow 

•	SUMO Traffic simulator

•	Traffic Control Interface (TraCI) – this is included with SUMO

## **Additional files for the traffic generation and intersection layout:**
•	Add.xml – This file for the induction loops and initialling the traffic light phases.

•	Rou.xml – This file is created when running the code. It is for the vehicle routes and the paths in the simulation.

•	Con.xml – This file is for the round connections in the simulations.

•	Edg.xml – This is for the lanes.

•	Nod.xml – This is for the state and end points for the roads.

•	Net.xml – This is a configuration file to combine all the above files and create the road network.

•	Netccfg – This is a sumo network configuration.

•	Sumocfg – This is GUI file for the simulation


## **References:** 
1.	Vidali A, Crociani L, Vizzari G, Bandini,S, (2019). Bandini. A Deep Reinforcement Learning Approach to Adaptive Traffic Lights Management [cited 23 August 2019]. Available from: http://ceur-ws.org/Vol-2404/paper07.pdf
2.	Gao J, Shen Y, Liu J, Ito M and Shiratori N. Adaptive Traffic Signal Control: Deep Reinforcement Learning Algorithm with Experience Replay and Target Network. [Internet]. Arxiv.org. 2019 [cited 28 June 2019]. Available from: https://arxiv.org/pdf/1705.02755.pdf
3.	Liang X, Du X, Wang G, Han Z. (2018). Deep Reinforcement Learning for Traffic Light Control in Vehicular Networks. [cited 10 July 2019]. Available from: https://www.researchgate.net/publication/324104685_Deep_Reinforcement_Learning_for_Traffic_Light_Control_in_Vehicular_Networks
4.	DLR - Institute of Transportation Systems - Eclipse SUMO – Simulation of Urban MObility [Internet]. Dlr.de. 2019 [cited 10 July 2019]. Available from: https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/





