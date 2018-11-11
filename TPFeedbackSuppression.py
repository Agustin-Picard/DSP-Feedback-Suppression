import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


"""Developed by Agustin M. Picard for a DSP course"""

class LMSFilter(object):
    """Least Mean Squares adaptive filter with N taps and mu step size"""
    def __init__(self, taps, mu):
        """Initialization of the LMS filter
        :param taps: filter length
        :param mu: step-size of the gradient descent
        """
        self.taps = taps
        self.mu = mu
        self.wk = np.zeros(taps)

    def filter(self, data, r):
        """Filters the data using an LMS adaptive filter scheme
        :param data: data to be filtered
        :param r: feedback signal
        :return s: filtered signal
        """
        s = np.zeros(data.shape)

        for i in range(data.shape[0] - self.taps):
            err = data[i:i+self.taps] - np.dot(r[i:i+self.taps].T, self.wk)
            self.wk += np.dot(self.mu, np.dot(r[i:i+self.taps].T, err))
            s[i:i+self.taps] = np.dot(self.wk.T, data[i:i+self.taps])
        return s


class NLMSFilter(object):
    """Similar to the LMS filter but here the step-size is a
    normalization of the feedback signal
    Not used because it doesn't converge for this problem
    """
    def __init__(self, taps):
        """Initialization of the NLMS filter
        :param taps: filter length
        """
        self.taps = taps
        self.wk = np.zeros((taps, 1))

    def filter(self, data, r):
        s = np.zeros(data.shape)
        for i in range(data.shape[0] - self.taps):
            err = data[i:i+self.taps] - np.dot(r[i:i+self.taps].T, self.wk)
            mu = 1 / (np.linalg.norm(r[i:i+self.taps])**2)
            self.wk = self.wk + np.dot(mu, np.dot(r[i:i + self.taps].T, err))
            s[i:i+self.taps] = np.dot(self.wk.T, data[i:i+self.taps])

        return s


class RLSFilter(object):
    """Recursive Least Squares adaptive filter with N taps and parameters delta and lambda"""
    def __init__(self, taps, delta, lmb):
        """Initialization step of the RLS filter
        :param taps: filter length
        :param delta: covariance initialization coefficient
        :param lmb: memory parameter
        """
        self.taps = taps
        self.lmb = lmb
        self.delta = delta
        self.wk = np.zeros(taps)

    def filter(self, data, r):
        """Filters the data with an RLS scheme
        :param data: data to be filtered
        :param r: feedback signal
        :return: filtered signal
        """
        s = np.zeros(data.shape)
        R_inv = self.delta * np.eye(self.taps)

        for i in range(data.shape[0] - self.taps):
            err = data[i:i+self.taps] - np.dot(r[i:i+self.taps], self.wk)
            K = np.dot(R_inv, r[i:i+self.taps]) / \
                (self.lmb + np.dot(r[i:i+self.taps].T, np.dot(R_inv, r[i:i+self.taps])))
            R_inv = (R_inv - K * np.dot(r[i:i+self.taps], R_inv)) / self.lmb
            self.wk += K * err
            s[i:i + self.taps] = err

        return s


def import_audio(filename):
    """Imports a .wav file as a scaled float numpy array
    :param filename: audio's filename
    :return fs: data rate
    :return audio: audio signal
    """
    (fs, audio) = scipy.io.wavfile.read(filename)
    audio = np.asarray(audio, dtype=float)
    audio /= np.linalg.norm(audio)
    return fs, audio


def compare_spectrograms(audio, audio_lms, audio_rls, fs):
    """Plots the spectrogram of the audio and
    the filtered audio signals in a subplot"""
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.specgram(x=audio, Fs=fs)
    plt.subplot(1, 3, 2)
    plt.title('LMS')
    plt.specgram(x=audio_lms, Fs=fs)
    plt.subplot(1, 3, 3)
    plt.title('RLS')
    plt.specgram(x=audio_rls, Fs=fs)
    plt.show()


def gen_feedback_signal(data, lag):
    """Generates the lagged feedback signal"""
    return np.concatenate((np.zeros(lag), np.roll(data, lag)[lag:]))


def main():
    # Define some parameters
    lag_lms = 64
    lag_rls = 46
    taps_lms = 24
    taps_rls = 67
    mu_lms = 0.005
    delta_rls = 0.0002
    lambda_rls = 0.9999

    # import the audio from the file
    (fs, audio) = import_audio('charla_salon1_con_mic_y_ampli.wav')

    # Initialize the filters
    lms_filter = LMSFilter(taps=taps_lms, mu=mu_lms)
    rls_filter = RLSFilter(taps=taps_rls, delta=delta_rls, lmb=lambda_rls)

    # Generate the feedback for each filter
    r_lms = gen_feedback_signal(audio, lag_lms)
    r_rls = gen_feedback_signal(audio, lag_rls)

    # Do the filtering for the three of them
    audio_lms = lms_filter.filter(audio, r_lms)
    audio_rls = rls_filter.filter(audio, r_rls)

    # Draw the spectrograms to compare the results
    compare_spectrograms(audio, audio_lms, audio_rls, fs)

    # Export to wav to verify that the noise is gone
    scipy.io.wavfile.write('FilteredAudioLMS.wav', fs, audio_lms*1000)  # Gain to give it more volume
    scipy.io.wavfile.write('FilteredAudioRLS.wav', fs, audio_rls*10)


if __name__ == '__main__':
    main()
