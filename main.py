import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sound_function(time: np.double) -> np.double:
    '''' The test function for the unnoised sound signal
    '''
    sound = 0.1*np.sin(time) + 0.2*np.sin(0.1*time) + 0.6*np.sin(2*time) + 0.8*np.sin(3*time)
    return sound

def create_shifted_input_data(sound, tdl_size: int, measurement_size: int) -> np.ndarray:
    ''' Create the shifted input data.
    Output dataset:
        sound step  0
        sound step -1
        sound step -2
            ...
        sound step -n
        with n as tdl_size
    '''
    shifted_data = np.zeros((tdl_size+1, measurement_size))
    for i in range(tdl_size+1):
        shifted_data[i, tdl_size:] = np.roll(sound[tdl_size:], i)
    return shifted_data

def normalize_dataset(sound: np.ndarray) -> np.ndarray:
    ''' The function normalizes the sound between -1 and 1
    '''
    maximum = np.max(sound)
    minimum = np.min(sound)
    normalized_data = 2*sound/(maximum - minimum)
    normalized_data = normalized_data - np.mean(normalized_data)
    return normalized_data

if __name__ == "__main__":
    # Main factors
    sampling_size = 8000
    tdl_size = 6
    mu = 0
    sigma = 0.14
    split_size = 0.85
    
    # Create the noised signal
    noise = np.random.uniform(-sigma, sigma, sampling_size) + mu
    time = np.linspace(0, 100, sampling_size)
    sound = sound_function(time)
    sound = normalize_dataset(sound)
    noised_sound = sound + noise

    # Create the input and output data of the neural network
    # The inputs in the neural network are the n-last signals with n as the number of the tapped delays
    input_sound = create_shifted_input_data(sound, tdl_size, sampling_size)
    input_noised_sound = create_shifted_input_data(noised_sound, tdl_size, sampling_size)
    input_data = input_noised_sound[:, tdl_size:input_noised_sound.shape[1]-1]
    output_data = sound[tdl_size+1:]
    output_data = np.expand_dims(output_data, axis=1)

    # Split the data into train and test data
    train_input_data = (input_data.T)[:int(split_size*sampling_size), :]
    test_input_data = (input_data.T)[int(split_size*sampling_size)+1:, :]
    train_output_data = output_data[:int(split_size*sampling_size), :]
    test_output_data = output_data[int(split_size*sampling_size)+1:, :]

    # Create a simple feedforward neural network
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(1)
    ])

    # Train the neural network and predict a time series
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_input_data, train_output_data, epochs=20, validation_split=0.2, batch_size=10)
    predict_output_data = model.predict(test_input_data)

    # Compare the predicted data with the test data for the output
    plt.figure(figsize=(7, 9), dpi=120)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(test_output_data[:-1])), test_output_data[:-1], c='blue', label='Test Sound Signal')
    plt.xlabel('Time')
    plt.ylabel('Sound Signal')
    plt.ylim(-2, 2)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(test_input_data.shape[0]-1), test_input_data[:-1, 0], c='red', alpha=0.6, label='Noised Sound Signal')
    plt.plot(np.arange(len(predict_output_data[1:])), predict_output_data[1:], c='green', alpha=0.8, label='Predicted Sound Signal')
    plt.xlabel('Time')
    plt.ylabel('Sound Signal')
    plt.ylim(-2, 2)
    plt.legend()
    plt.savefig('output.png')