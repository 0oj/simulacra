Our rocket. Is it god?

As a simulation, we have to model mimic how physics works.
We need to know the 'true' position of the rocket.
The rocket itself should not have access to its true position.

There are two rockets:
1. the onboard flight computer that relies on sensor data and creates estimates.
2. the 'rocket' inside the simulation which will hold information like its true position.
Rocket 1. might contain a parameter called 'position' (rather, 'position_estimate').
But Rocket 2. shall hold information for the simulation god.

This is a naming problem. A knife cuts a boundary.
Some information belongs to the omniscient god that runs the simulation,
and some belongs to the poor ignorant rocket that can only make guesses.

Since we must first invent the universe before it is inhabited by the concious,
this file will be running in godmode. Let there be light, later.

Then classes here are just a method to 'organise' information, but not organise 'Information'.
God or not, tells us whether this information is privileged and true.
But we must still expose an interface to the rocket's computer, give it its eyes, and hear its cries.

A scenario will be created by a simulation engineer looking to test something.
They will need to input a specific rocket. This rocket will have properties.
These properties will be physical properties like length and weight.
And eventually maybe even methods to import from an .stl file.
So a scenario class should initialise a scenario with a specific rocket.
So you have to create a rocket first. A rocket should then expect information:
- what is your size? - what is your weight? what is your drag coeff?
Does this need to be in God? We are only just instantiating a rocket?
The scenario would also ask for events, and initialisations of the world.
- what is the gravity? - what is the wind? - etc.?
And it will contain events I want to happen. Like a Setpoint(timestamp, position).
This is a control message that should reach the rocket's control system.
It should basically have access to everything in the world. Like:
[timestamp, (world) -> world.gravity = 0] # turn gravity off at timestamp.

The distiction here is can be defined between the 'GOD' and the 'prisoner'.
A case for the renaming of this project from Simulacra to Allegory/Cave.

Another name is Episteme. Tho it is gay.
