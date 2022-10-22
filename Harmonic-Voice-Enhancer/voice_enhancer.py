'''importing required libraries'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

'''Reading the original.wav file'''
sampling_rate,data_array=wavfile.read('./original.wav')

'''Generating the time axis (start,stop,num)'''
time=np.linspace(0,len(data_array)/sampling_rate,len(data_array))

class voice_enhancer():
    def time_domain_analysis(self):
        '''Printing the time domain signal'''
        
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1280x720
        plt.plot(time,data_array)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Time domain signal of original.wav')
        plt.show()
    
    def normalised_signal(self):
        '''Printing the normalized time domain signal'''
        
        peak = max(abs(data_array))
        norm_data = data_array*(1/peak)
        quant_min = -1
        quant_max = 1
        q_level = 2**16
        x_normalize = (norm_data-quant_min) * (q_level-1) / (quant_max-quant_min)
        x_normalize[x_normalize > q_level - 1] = q_level - 1
        x_normalize[x_normalize < 0] = 0
        x_normalize_quant = np.around(x_normalize)
        x_quant = (x_normalize_quant) * (quant_max-quant_min) / (q_level-1) + quant_min
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1280x720
        plt.plot(time, x_quant)
        plt.xlabel('Time [s]')
        plt.ylabel('Normalised amplitude')
        plt.title('Quantised version of original.wav')
        plt.show()
        return print('Normalised array: {0} \n Original array: {1}'.format(x_quant,norm_data))

      
    
    def frequency_domain_analysis(self):    
        '''FFT of the original.wav data'''
        
        self.data_fft=np.fft.fft(data_array)
        fftd=abs(self.data_fft)                                                #Using only the absolute values
        self.x=fftd[0:int((len(fftd)/2)-1)]                                    #Using only the half range of the frequency axis
        self.freqaxis=np.linspace(0,sampling_rate/2,len(self.x))               #Range of  the frequency axis   
        
        '''Printing the time domain signal'''
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1280x720                           
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.title('Frequency domain signal of original.wav')
        plt.xscale('log')                                                      #Plotting the x axis in logarithmic scale
        plt.plot(self.freqaxis,20*np.log10(self.x/len(data_array)))            #Converting the amplitude to dB
        plt.show()
        
        ''''Setting the position of the frequency ranges of interest'''
        self.n1=int(len(self.data_fft)/sampling_rate*0.1)                      #index position for 0.1Hz
        self.n2=int(len(self.data_fft)/sampling_rate*99)                       #index position for 99Hz
        self.n3=int(len(self.data_fft)/sampling_rate*8001)                     #index position for 8001Hz
        self.n4=int(len(self.data_fft)/sampling_rate*22050)                    #index position for 22050Hz
        self.k1=int(len(self.data_fft)/sampling_rate*900)                      #index position for 900Hz
        self.k2=int(len(self.data_fft)/sampling_rate*8000)                     #index position for 8000Hz        
    
    def noise_reduction(self):
        
        '''Reducing the amplitude of the signal in the frequency ranges 0.1-99 Hz and 8001Hz to 22050Hz'''
        self.data_fft[self.n3:self.n4]=self.data_fft[self.n3:self.n4]*0.01
        self.data_fft[int(len(self.data_fft)-self.n4):int(len(self.data_fft)-self.n3)]=self.data_fft[int(len(self.data_fft)-self.n4):int(len(self.data_fft)-self.n3)]*0.01
        self.data_fft[self.n1:self.n2]=self.data_fft[self.n1:self.n2]*0.01
        self.data_fft[int(len(self.data_fft)-self.n2):int(len(self.data_fft)-self.n1)]=self.data_fft[int(len(self.data_fft)-self.n2):int(len(self.data_fft)-self.n1)]*0.01
        
    def harmonic_amplification(self):
        
        '''Amplification of the harmonics'''
        self.data_fft[self.k1:self.k2]=self.data_fft[self.k1:self.k2]*10      
        self.data_fft[int(len(self.data_fft)-self.k2):int(len(self.data_fft)-self.k1)]=self.data_fft[int(len(self.data_fft)-self.k2):int(len(self.data_fft)-self.k1)]*10
    
    def ifft_analysis(self):
        
        '''Printing the enhanced frequency domain signal'''
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1280x720
        plt.plot(self.freqaxis,20*np.log10(self.data_fft[0:int(len(self.data_fft)/2)-1]/len(self.data_fft)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.title('Improved frequency domain')
        plt.xscale('log')                                                      #Plotting the x axis in logarithmic scale
        plt.show()
        
        '''IFFT of the enhanced data'''
        enhanced=np.fft.ifft(self.data_fft)
        clr=np.real(enhanced)                                                  #Extracting the real part
        audio = clr.astype(np.int16)                                           #Converting to 16 bit data
        
        '''Printing the enhanced time domain signal'''
        plt.figure(figsize=(13.33,7.5))                                        #setting the output picture size to 1280x720
        plt.plot(time,clr)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Improved time domain signal/improved.wav')
        plt.show()
        return wavfile.write('improved.wav',sampling_rate,audio)               #Saving the enhanced audio file
   
    def fundamental_frequency(self):
        
        '''Printing the fundamental frequency and its amplitude'''
        x_db=20*np.log10(self.x/len(data_array))
        xmax=max(x_db)
        max_amp_freq=self.freqaxis[np.argmax(self.x)]    
        return print('\n The fundamental frequency of the signal is {1}Hz and it\'s amplitude is {0}dB'.format(round(xmax,3),round(max_amp_freq,3)))

VE = voice_enhancer()
Time_domain = VE.time_domain_analysis()
Normalised_signal = VE.normalised_signal()
Frequency_domain = VE.frequency_domain_analysis()
Noise_reduction = VE.noise_reduction()
Amplification = VE.harmonic_amplification()
IFFT_analysis = VE.ifft_analysis()
Fundamental_Frequency = VE.fundamental_frequency()
 