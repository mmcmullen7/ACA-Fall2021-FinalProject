ABOUT THE DATASET:

This dataset contains a large library of 2,656 unique expressive drum samples recorded in a standard drum kit. 
Due to 4 recording techniques per sample, there are 10,624 total samples. Below is a list of the included instruments and their parameters. The many dimensions of dataset labels are embedded in the folder structure and filenames.

Filenames look like this:
	TEXT_Text_Text_####.#.wav

The positions correspond to:
	{MixType}_{InsrumentType}_{Articularion Name}_#{Snare on/off 0/1} #{ intensity } #{strike position} #{stick height}. #{example}.wav

For example:
	DI_Rack Tom_Rim_0234.2.wav corresponds to
	- a direct mic'ed recording
	- rack tom sample 
	- a rim shot articulation
	- that was recorded without the snares on the snare drum
	- at a medium intensity (intensity is staccato vs legato in this case)
	- halfway between the center and the edge of the drum
	- from a height of 36cm (~ 14 inches) above the drum.



MIX TYPE

All of the drums are recorded in the studio with a snare drum present with the snares both on and off to add realism to the sample. 
The cymbals are recorded in isolation. Each sample was recorded with both direct (close) and indirect (room) mic’ing techniques. 
Both a mono and stereo mix of each sample is included as well as the individual direct and indirect raw recordings. The recording type prefixes are as follows:

DI - Direct
IN - Indirect
MN - Mono Mix
ST - Stereo Mix



INSTRUMENT TYPES

Long Kick - deep kick drum
Long Kick SNoff - deep kick recorded without snare drum resonance
Dead Kick - shallow dampened kick drum
Dead Kick SNoff - shallow dampened recorded without snare drum resonance
Rack Tom - 12" rack tom
Rack Tom SNoff - 12" rack tom recorded without snare drum resonance
Floor Tom - 16" floor tom
Floor Tom SNoff - 16" floor tom recorded without snare drum resonance
Bright Crash - 16" medium crash
Dark Crash - 18" thin crash
Hi-Hat - 14" medium hi-hat
Ride - 20" medium ride



What is ARTICULATION?
	
	Articulation is the method of striking the instrument. 
	
	For Drums with Sticks:
		- strike : standard hit of the drum
		- buzz stroke : press the stick into the head to achieve multiple bounces
		- rim : strike of both the rim and drum simultaneously (rim shot). The edge case is just the rim.
		- cross stick : strike the rim with the butt of the stick while keeping the tip on the head.
	For Bass Drum
		- Press : after the pedal mallet strikes the drum, allow it to rest against the head and dampen the sound
		- Release : after the pedal mallet strikes the drum, release it quickly allowing the drum to vibrate
	For Cymbals
		- Tip : strike cymbal with tip of stick.
		- Crash : strike cymbal with the shaft of stick.
		- Clamp : strike cymbal with shaft of stick and choke with hand to dampen.
	For Hi-hat
		- Tip : strike cymbal with tip of stick.
		- Foot : use the foot pedal to close the two cymbals together.



What is INTENSITY?
	
	Intensity is the strength of the articulation effect in terms of instrument and articulation.
	
	For Snare and Toms:
		- strike : staccato vs legato in context of drums. Basically it corresponds to the heaviness of stroke.
		- Buzz : the short vs long nature of the buzz
	For Bass Drum
		- N/A
	For Crash and Ride Cymbals
		- N/A
	For Hi-hat
		- tip : Foot pedal pressure.
		- foot : N/A



What is STRIKE POSITION?

	Strike position is the radial distance of the strike from the center of the instrument in terms of instrument and articulation
	
	For Drums with Sticks:
		- strike : center to edge
		- buzz stroke : center to edge
		- rim : center to edge to hitting the rim only
		- cross stick : tip against the rim vs tip halfway
	For Bass Drum
		- N/A
	For Cymbals
		- Tip : bell to edge
		- Crash : bell to edge to shaft on rim of cymbal
		- Clamp : bell to edge to shaft on rim of cymbal
	For Hi-hat
		- Tip : bell to edge
		- Foot : heaviness of foot release



What is HEIGHT?
	
	Height is the physical distance from which the drum is struck. Corresponds closely with dynamics.
	
	For Snare, Toms, Cymbals:
		- height from which the instrument is struck
	For Bass Drum
		- heaviness of the foot press.
	For Hi-hat
		- For tip:
			- height from which the instrument is struck
		- For Foot:
			- heaviness of the foot press



RANGE of EXPRESSION PARAMETERS
(see ArticulationRange.tsv)
Cymbals
	- Bright Crash
		- Clamp
			Intensity: 1	Strike Position: 1	Stick Height: 1-4	Examples: 4
		- Crash
			Intensity: 1	Strike Position: 1	Stick Height: 1-4	Examples: 4
		- Tip
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
	- Dark Crash
		- Clamp
			Intensity: 1	Strike Position: 1	Stick Height: 1-4	Examples: 4
		- Crash
			Intensity: 1	Strike Position: 1	Stick Height: 1-4	Examples: 4
		- Tip
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
	- HiHat
		- Tip
			Intensity: 1-5	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Foot
			Intensity: 1	Strike Position: 1-5	Stick Height: 1-4	Examples: 4
	- Ride
		- Tip
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4

Kick Drum
	- Long Kick
		- Press
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
		- Release
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
	- Dead Kick
		- Press
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
		- Release
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
	- Long Kick SNoff
		- Press
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
		- Release
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
	- Dead Kick SNoff
		- Press
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6
		- Release
			Intensity: 1	Strike Position: 1	Stick Height: 1-5	Examples: 6

Tom Drums
	- Rack Tom
		- Buzz
			Intensity: 1-2	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
		- Rim
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Strike
			Intensity: 1-3	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
	- Rack Tom SNoff
		- Buzz
			tensity: 1-2	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
		- Rim
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Strike
			Intensity: 1-3	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
	- Floor Tom
		- Buzz
			tensity: 1-2	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
		- Rim
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Strike
			Intensity: 1-3	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
	- Floor Tom SNoff
		- Buzz
			Intensity: 1-2	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
		- Rim
			Intensity: 1	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Strike
			Intensity: 1-3	Strike Position: 1-2	Stick Height: 1-4	Examples: 4

Snare
	- Snare
		- Buzz
			Intensity: 1-3	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Rim
			Intensity: 1-3	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Strike
			Intensity: 1-3	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Cross Stick
			Intensity: 1-3	Strike Position: 1-2	Stick Height: 1-4	Examples: 4
	- Snare SNoff
		- Buzz
			Intensity: 1-3	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Rim
			Intensity: 1-3	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Strike
			Intensity: 1-3	Strike Position: 1-3	Stick Height: 1-4	Examples: 4
		- Cross Stick
			Intensity: 1-3	Strike Position: 1-2	Stick Height: 1-4	Examples: 4


WHO CREATED IT?

This dataset was collected as a joint effort between students in the Music and Entertainment Technology Lab in the ECE Dept. in the Drexel University College of Engineering (COE) and the Music Industry Department in the Drexel University College of Media Arts and Design (COMAD). Drexel University has since released the rights to the data and have given them to Xpressive Instruments LLC, a small start up that was founded by a few members of the original project. The data is free to use and distribute for educational and research purposes only (see license below).

Any published or derivative works that use the data or any portion of the data (audio files, audio filenames, and audio file descriptions) must cite the following paper:
Matthew Prockup, Erik M. Schmidt, Jeffrey J. Scott, and Youngmoo E. Kim. "Toward Understanding Expressive Percussion Through Content Based Analysis." In ISMIR, pp. 143-148. 2013.


LICENSE:

The data (audio files, audio filenames, audio file descriptions) contained in this dataset (MDLib2.2) is intended for educational and research purposes only. The data or any portion of the data may not be reproduced, duplicated, copied, sold, resold, or otherwise exploited for any commercial purpose that is not expressly authorized by Xpressive Instruments LLC. Except as expressly provided herein, Xpressive Instruments LLC does not grant you any express or implied rights under any patents, copyrights, trademarks, trade secret, or other intellectual or industrial property rights.

Any published or derivative works that use the data or any portion of the data (audio files, audio filenames, and audio file descriptions) must cite the following paper:
Matthew Prockup, Erik M. Schmidt, Jeffrey J. Scott, and Youngmoo E. Kim. "Toward Understanding Expressive Percussion Through Content Based Analysis." In ISMIR, pp. 143-148. 2013.

This dataset is provided by Xpressive Instruments LLC on an "as is" basis. Xpressive Instruments LLC makes no representations or warranties of any kind, express or implied, as to the information, content, materials, products or services included in this dataset. To the full extent permissible by applicable law, Xpressive Instruments LLC disclaims all warranties, express or implied, including, but not limited to, implied warranties of merchantability and fitness for a particular purpose and non-infringement. Xpressive Instruments LLC does not warrant the accuracy or completeness of the information contained within this dataset and in no event shall Xpressive Instruments LLC be liable under any theory of law, for any indirect, incidental, punitive or consequential damages.