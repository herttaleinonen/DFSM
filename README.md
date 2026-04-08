# The Dynamic Foveated Search Model 

The model receives three fixed inputs during each search trial: (1) sensory sensitivity estimates from the visibility task, expressed as a discriminability function d^' (e,v)  over retinal eccentricity e and stimulus velocity v; (2) the positions of all Gabor stimuli over time; and (3) the human observer’s gaze trajectory. (A) At each time step, the model computes the retinal eccentricity of each object relative to gaze and converts it into momentary sensory sensitivity using the calibrated d^' (e,v)  function. (B) Given this sensitivity, the model samples noisy sensory evidence for each object. These samples are transformed into object specific log likelihood ratios quantifying support for the hypothesis that the object is the target. (C) Evidence is then accumulated over time and pooled across objects to form a global decision variable representing support for target presence anywhere in the display. (D) After each update, the decision variable is compared against upper and lower bounds. Crossing the upper bound triggers a “target present” response, while crossing the lower bound triggers a “target absent” response; otherwise, accumulation continues until stimulus terminates. The model outputs predicted sensitivity (d^') and reaction times for both target present and target absent conditions.

# Running the model

Run main.py with selected settings.  



