# The experimental setup

Set of experiments developed to study visual search in dynamic environments. Includes dynamic visual search tasks (a) and dynamic visibility tasks (b). 

<a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/experimental_setup/images/Picture_1.png" height="448" width="448" ></a> <a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/experimental_setup/images/Picture_2.png" height="348" width="348" ></a>


## Requirements 

- python3 (3.10 or newer)
- PsychoPy (2024.2.4)
- EyeLink 1000 Plus + pylink to run with eye tracking
- RB-530 or similar + pyxid2 to run with Cedrus response box.

## Running 

Run main.py through PsychoPy Coder. 

Number of trials and object speed for the search task can be altered by modifying the file `config.py`.

To change the object speed in the visibility task, modify the file `visibility_dynamic.py`.
