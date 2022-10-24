'''importing required libraries'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks 


class vowels():

    def voweldetector(self,audio):
        '''Reading the original.wav file'''
        
        sampling_rate,data_array=wavfile.read(audio)
        
        '''Generating the time axis (start,stop,num)'''
        
        time=np.linspace(0,len(data_array)/sampling_rate,len(data_array))
        
        '''Printing the time domain signal'''
        
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1920x1080
        plt.plot(time,data_array)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Time domain signal of {0}'.format(audio))
        plt.show()
    

        '''FFT of the original.wav data'''
        
        data_fft=np.fft.fft(data_array)
        fftd=abs(data_fft)                                                     #Using only the absolute values
        x=fftd[0:int((len(fftd)/2)-1)]                                         #Using only the half range of the frequency axis
        freqaxis=np.linspace(0,sampling_rate/2,len(x))                         #Range of  the frequency axis
        
        '''Printing the frequency domain signal'''    
                     
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1920x1080                            
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Frequency domain signal of {0}'.format(audio))
        plt.xscale('log')                                                      #Plotting the x axis in logarithmic scale
        plt.plot(freqaxis,(x/len(data_array)))                                 #Converting the amplitude to dB
        plt.show()
         
        '''Printing the fundamental frequency and its amplitude''' 
        
        x_db = 20*np.log10(x/len(data_array))
        xmax = max(x_db)
        max_amp_freq=freqaxis[np.argmax(x)] 
        
        """Finding the index of Fundamental or 1st Formant frequency"""
        
        index=np.unravel_index(np.argmax(x_db,axis=None),x_db.shape)
        k1=index[0]
        
        """Finding the 2nd Formant peak and its frequency"""
        plt.figure(figsize=(13.33,7.5))  
        nor = x[k1:]/len(data_array)                                        #slicing the fft signal from the fundamental frequency (k1) to the end
        new_freqaxis=freqaxis[k1:]                                          #slicing the frequency axis to match the sliced fft data
        peak_data=find_peaks(nor,height=10,prominence=4, distance =500)     #finding the peaks in the sliced data, output is a dictionary of index, peak values and associated frequencies 
        peak, _ = find_peaks(nor,height=10,prominence=4, distance =500)     #taking only the index data from peak_data dictionary
        sec_formant_freq = new_freqaxis[peak_data[0]]
        plt.plot(nor)
        plt.plot(peak, nor[peak], "*")
        plt.legend(['Frequency response from 1st formant/fundamental frequency', 'Maxima of 2nd Formant followed by other formant peaks if present'])
        plt.title('Frequency domain signal only from first formant/fundamental frequency of {0}'.format(audio))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.show()
        
        
        '''Determining the vowel based on the main Formant (fundamental) and 2nd Formant frequencies'''
        if (max_amp_freq>=200 and max_amp_freq<=400):
            return print('\n The vowel is \"u\" with main formant(fundamental) frequency of {1}Hz and amplitude {0}dB with no 2nd formant frequency'.format(round(xmax,3),round(max_amp_freq,3)))
        elif (max_amp_freq>400 and max_amp_freq<=600 and sec_formant_freq[0] <= 1000):  
            return print('\n The vowel is \"o\" with main formant(fundamental) frequency of {1}Hz and amplitude {0}dB with 2nd formant frequency at {2}Hz'.format(round(xmax,3),round(max_amp_freq,3), round(sec_formant_freq[0],3) ))
        else:
            return print('\n The fundamental frquency is {0}Hz with 2nd Formant frequency of {1}Hz, therefore it is a consonant and not \'o\' or \'u\'.'.format(round(max_amp_freq,3),round(sec_formant_freq[0],3)))

        
VE = vowels()                                                                  #Instantiating class vowels
Vowel1= VE.voweldetector('./vowel1.wav')
Vowel2= VE.voweldetector('./vowel2.wav')
consonant= VE.voweldetector('./consonant_z.wav')
