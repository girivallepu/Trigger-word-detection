
                                               Trigger Word Detection
As part of applying deep learning to speech recognition.  constructed a speech dataset and implemented an algorithm for trigger word detection (sometimes also called keyword detection, or wakeword detection). Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word. 
In our implementation, trigger word is "Activate." Every time it hears you say "activate," it will make a "chiming" sound. 

1 - Data synthesis: Creating a speech dataset
Let's start by building a dataset for trigger word detection algorithm. A speech dataset should ideally be as close as possible to the application run it on. In this case, we like to detect the word "activate" in working environments (library, home, offices, open-spaces ...). You thus need to create recordings with a mix of positive words ("activate") and negative words (random words other than activate) on different background sounds. 

1.1 - Listening to the data
Dataset has been created and it includes people speaking in a variety of accents and recorded in background noises, as well as snippets of audio of people saying positive/negative words at libraries, cafes, restaurants, homes and offices. In the raw_data directory, you can find a subset of the raw audio files of the positive words, negative words, and background noise. We use these audio files to synthesize a dataset to train the model. The "activate" directory contains positive examples of people saying the word "activate". The "negatives" directory contains negative examples of people saying random words other than "activate". There is one word per audio recording. The "backgrounds" directory contains 10 second clips of background noise in different environments.
We used these three type of recordings (positives/negatives/backgrounds) to create a labelled dataset.

1.2 - From audio recordings to spectrograms
What really is an audio recording? A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear also perceives as sound. You can think of an audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone. We will use audio sampled at 44100 Hz (or 44100 Hertz). This means the microphone gives us 44100 numbers per second. Thus, a 10 second audio clip is represented by 441000 numbers (= 10×44100). 
It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said. In order to help our sequence model more easily learn to detect triggerwords, we will compute a spectrogram of the audio. The spectrogram tells us how much different frequencies are present in an audio clip at a moment in time. 
Lets see an example. 
x = graph_spectrogram("audio_examples/example_train.wav")
 
Figure 1: Spectrogram of an audio recording, where the color shows the degree to which different frequencies are present (loud) in the audio at different points in time. Green squares means a certain frequency is more active or more present in the audio clip (louder); blue squares denote less active frequencies. 
The graph above represents how active each frequency is (y axis) over a number of time-steps (x axis). 
The dimension of the output spectrogram depends upon the hyperparameters of the spectrogram software and the length of the input. We will be working with 10 second audio clips as the "standard length" for our training examples. The number of timesteps of the spectrogram will be 5511. 
Note that even with 10 seconds being our default training example length, 10 seconds of time can be discretized to different numbers of value. We've seen 441000 (raw audio) and 5511 (spectrogram). In the former case, each step represents 10/441000≈0.000023 seconds. In the second case, each step represents 10/5511≈0.0018 seconds. 

For the 10sec of audio, the key values we will see are:
•	441000 (raw audio)
•	5511=Tx (spectrogram output, and dimension of input to the neural network). 
•	10000 (used by the pydub module to synthesize audio) 
•	1375=Ty (the number of steps in the output of the GRU). 
Note that each of these representations correspond to exactly 10 seconds of time. It's just that they are discretizing them to different degrees. All of these are hyperparameters and can be changed (except the 441000, which is a function of the microphone). We have chosen values that are within the standard ranges uses for speech systems. 
Consider the Ty=1375 number above. This means that for the output of the model, we discretize the 10s into 1375 time-intervals (each one of length 10/1375≈0.0072s) and try to predict for each of these intervals whether someone recently finished saying "activate." 
Consider also the 10000 number above. This corresponds to discretizing the 10sec clip into 10/10000 = 0.001 second itervals. 0.001 seconds is also called 1 millisecond, or 1ms. So when we say we are discretizing according to 1ms intervals, it means we are using 10,000 steps. 

1.3 - Generating a single training example
Because speech data is hard to acquire and label, you will synthesize your training data using the audio clips of activates, negatives, and backgrounds. It is quite slow to record lots of 10 second audio clips with random "activates" in it. Instead, it is easier to record lots of positives and negative words, and record background noise separately (or download background noise from free online sources). 

To synthesize a single training example, we will:
•	Pick a random 10 second background audio clip
•	Randomly insert 0-4 audio clips of "activate" into this 10sec clip
•	Randomly insert 0-2 audio clips of negative words into this 10sec clip

Because we had synthesized the word "activate" into the background clip, we know exactly when in the 10sec clip the "activate" makes its appearance. You'll see later that this makes it easier to generate the labels y⟨t⟩ as well. 
Use the pydub package to manipulate audio. Pydub converts raw audio files into lists of Pydub data structures. Pydub uses 1ms as the discretization interval (1ms is 1 millisecond = 1/1000 seconds) which is why a 10sec clip is always represented using 10,000 steps. 

Overlaying positive/negative words on the background:
Given a 10sec background clip and a short audio clip (positive or negative word), we need to be able to "add" or "insert" the word's short audio clip onto the background. To ensure audio segments inserted onto the background do not overlap, we will keep track of the times of previously inserted audio clips. You will be inserting multiple clips of positive/negative words onto the background, and you don't want to insert an "activate" or a random word somewhere that overlaps with another clip you had previously added. 
For clarity, when you insert a 1sec "activate" onto a 10sec clip of cafe noise, you end up with a 10sec clip that sounds like someone sayng "activate" in a cafe, with "activate" superimposed on the background cafe noise. we do not end up with an 11 sec clip. 

Creating the labels at the same time we overlay:

Recall also that the labels y⟨t⟩ represent whether or not someone has just finished saying "activate." Given a background clip, we can initialize y⟨t⟩=0 for all t, since the clip doesn't contain any "activates." 

When we insert or overlay an "activate" clip, we will also update labels for y⟨t⟩, so that 50 steps of the output now have target label 1. You will train a GRU to detect when someone has finished saying "activate". For example, suppose the synthesized "activate" clip ends at the 5sec mark in the 10sec audio---exactly halfway into the clip. Recall that Ty=1375, so timestep 687= int(1375*0.5) corresponds to the moment at 5sec into the audio. So, we will set y⟨688⟩=1. Further, you would quite satisfied if the GRU detects "activate" anywhere within a short time-internal after this moment, so we actually set 50 consecutive values of the label y⟨t⟩ to 1. Specifically, we have y⟨688⟩=y⟨689⟩=⋯=y⟨737⟩=1. 

This is another reason for synthesizing the training data: It's relatively straightforward to generate these labels y⟨t⟩ as described above. In contrast, if we have 10sec of audio recorded on a microphone, it's quite time consuming for a person to listen to it and mark manually exactly when "activate" finished. 

To implement the training set synthesis process, we used the following helper functions. All of these functions use a 1ms discretization interval, so the 10sec of audio is alwsys discretized into 10,000 steps. 
1.	get_random_time_segment(segment_ms) gets a random time segment in our background audio
2.	is_overlapping(segment_time, existing_segments) checks if a time segment overlaps with existing segments
3.	insert_audio_clip(background, audio_clip, existing_times) inserts an audio segment at a random time in our background audio using get_random_time_segment and is_overlapping
4.	insert_ones(y, segment_end_ms) inserts 1's into our label vector y after the word "activate"

The function get_random_time_segment(segment_ms) returns a random time segment onto which we can insert an audio clip of duration segment_ms. 

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

Next, suppose you have inserted audio clips at segments (1000,1800) and (3400,4500). I.e., the first segment starts at step 1000, and ends at step 1800. Now, if we are considering inserting a new audio clip at (3000,3600) does this overlap with one of the previously inserted segments? In this case, (3000,3600) and (3400,4500) overlap, so we should decide against inserting a clip here. 

For the purpose of this function, define (100,200) and (200,250) to be overlapping, since they overlap at timestep 200. However, (100,199) and (200,250) are non-overlapping. 

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap

Now, lets use the previous helper functions to insert a new audio clip onto the 10sec background at a random time, but making sure that any newly inserted segment doesn't overlap with the previous segments. 

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])

    return new_background, segment_time 

Finally, implement code to update the labels y⟨t⟩, assuming you just inserted an "activate." In the code below, y is a (1,1375) dimensional vector, since Ty=1375. 
If the "activate" ended at time step t, then set y⟨t+1⟩=1 as well as for up to 49 additional consecutive values. However, make sure you don't run off the end of the array and try to update y[0][1375], since the valid indices are y[0][0] through y[0][1374] because Ty=1375. So if "activate" ends at step 1370, you would get only y[0][1371] = y[0][1372] = y[0][1373] = y[0][1374] = 1

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1

    return y

Finally, you can use insert_audio_clip and insert_ones to create a new training example.

def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Make background quieter
    background = background - 20

    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1,Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    return x, y

Now you can listen to the training example you created and compare it to the spectrogram generated above. 

1.4 - Full training set
We've now implemented the code needed to generate a single training example. We used this process to generate a large training set.  We used set of training examples in XY_train.

1.5 - Development set
To test our model, we recorded a development set of 25 examples. While our training data is synthesized, we want to create a development set using the same distribution as the real inputs. Thus, we recorded 25 10-second audio clips of people saying "activate" and other random words, and labeled them by hand. 

2 - Model
Now that we've built a dataset, lets write and train a trigger word detection model! 

The model will use 1-D convolutional layers, GRU layers, and dense layers. Let's load the packages that will allow you to use these layers in Keras. 

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

2.1 - Build the model
Here is the architecture we will use. Below is the model  One key step of this model is the 1D convolutional step It inputs the 5511 step spectrogram, and outputs a 1375 step output, which is then further processed by multiple layers to get the final Ty=1375 step output. This layer extracts low-level features and then possibly generating an output of a smaller dimension. 
Computationally, the 1-D conv layer also helps speed up the model because now the GRU has to process only 1375 timesteps rather than 5511 timesteps. The two GRU layers read the sequence of inputs from left to right, then ultimately uses a dense+sigmoid layer to make a prediction for y⟨t⟩. Because y is binary valued (0 or 1), we use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said "activate."
Note that we use a uni-directional RNN rather than a bi-directional RNN. This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said. If we used a bi-directional RNN, we would have to wait for the whole 10sec of audio to be recorded before we could tell if "activate" was said in the first second of the audio clip. 
Implementing the model can be done in four steps:
Step 1: CONV layer. Use Conv1D() to implement this, with 196 filters, a filter size of 15 (kernel_size=15), and stride of 4.
 Step 2: First GRU layer. To generate the GRU layer, use:
X = GRU(units = 128, return_sequences = True)(X)
Setting return_sequences=True ensures that all the GRU's hidden states are fed to the next layer. Remember to follow this with Dropout and BatchNorm layers. 
Step 3: Second GRU layer. This is similar to the previous GRU layer (remember to use return_sequences=True), but has an extra dropout layer. 
Step 4: Create a time-distributed dense layer as follows: 
X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

This creates a dense layer followed by a sigmoid, so that the parameters used for the dense layer are the same for every time step. 

def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape = input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                  # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)              # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)              # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)

    return model  
    
Let's print the model summary to keep track of the shapes.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 5511, 101)         0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1375, 196)         297136    
_________________________________________________________________
batch_normalization_1 (Batch (None, 1375, 196)         784       
_________________________________________________________________
activation_1 (Activation)    (None, 1375, 196)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1375, 196)         0         
_________________________________________________________________
gru_1 (GRU)                  (None, 1375, 128)         124800    
_________________________________________________________________
dropout_2 (Dropout)          (None, 1375, 128)         0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 1375, 128)         512       
_________________________________________________________________
gru_2 (GRU)                  (None, 1375, 128)         98688     
_________________________________________________________________
dropout_3 (Dropout)          (None, 1375, 128)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 1375, 128)         512       
_________________________________________________________________
dropout_4 (Dropout)          (None, 1375, 128)         0         
_________________________________________________________________
time_distributed_1 (TimeDist (None, 1375, 1)           129       
=================================================================
Total params: 522,561
Trainable params: 521,657
Non-trainable params: 904
_________________________________________________________________

The output of the network is of shape (None, 1375, 1) while the input is (None, 5511, 101). The Conv1D has reduced the number of steps from 5511 at spectrogram to 1375. 

2.2 - Fit the model
Trigger word detection takes a long time to train.  We used already trained a model for about 3 hours on a GPU using the architecture you built above, and a large training set of about 4000 examples. 
model = load_model('./models/tr_model.h5')
We trained the model further, using the Adam optimizer and binary cross entropy loss, as follows. This will run quickly because we are training just for one epoch and with a small training set of 26 examples. 
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model.fit(X, Y, batch_size = 5, epochs=1)

2.3 - Test the model
We tested the model on DEV and got slightly over 90% accuracy. 

3 - Making Predictions
Now that we have built a working model for trigger word detection, let's use it to make predictions. 

3.3 - Test on dev examples
We have tested to check how our model performs on two unseen audio clips from the development set and we can hear it “Chime” when activate word is detected.

Here's what we should remember:
•	Data synthesis is an effective way to create a large training set for speech problems, specifically trigger word detection. 
•	Using a spectrogram and optionally a 1D conv layer is a common pre-processing step prior to passing audio data to an RNN, GRU or LSTM.
•	An end-to-end deep learning approach can be used to built a very effective trigger word detection system. 
