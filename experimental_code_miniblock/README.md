# DS5 controller #

Control of DS5 Bipolar Constant Current Stimulator via Matlab, Psychtoolbox and Data Acquisition Toolbox

## General ##

- Stimulation intensity relates to adjusted voltage and current (e.g., 10V:10mA -> signal of 5 relates to 5 mA)
- Analog signal usually variation of voltage
- DS5 current and voltage output have to multiplied by 10 (1V signal equals 10mA resp. 10V)
- If input positive, then red output is positive, so current flows from black to red
- IMPORTANT: Wait until output is done "talking" (e.g., trigger(ao); wait(ao,1000);)

### Latency ###

- DS5 produces no latency (comparison of direct input of analog output vs. DS5 current input)
- 2 analog output channels are triggered without latency (comparison direct inputs of analog outputs)

## Hardware ##

- Digitimer DS5
- Data Translation DT9812 USB module *or* National Instrument USB 6229

## System requirements ##

Determined by Data Translation's DAQ Adaptor for MATLAB (http://www.datatranslation.com/cd/omni/details.asp?cid=2&pid=12)

- Windows
- Matlab 32-bit
- Data Acquisition Toolbox 32-bit (http://de.mathworks.com/products/daq/)
- Psychotoolbox

### Data Translation software ###

- setup.EXE - DAQ Adaptor for Matlab Setup program
- SetupOEM.exe - Data Translation Open Layers (OEM)

### Installation ###

- install DT OEM
- install DT DAQ Adaptor for Matlab
- daqregister 'C:\Program Files (x86)\Data Translation\DAQ Adaptor for MATLAB\Dtol.dll'
- (move 'inpout32.dll' to 'C:\windows\system32')
- Download and copy the inpout32a.dll file to the C:\windows\sysWOW64 directory

### Linux or Mac OS compatibility ###

Unfortunately, there are no Linux or Mac OS drivers for the DT or NI USB modules.

I found only something for the DT9812:

The Comedi project includes a Linux driver for the DT9812 Waveform-Generator (w/o bulk transfer) and it seems to be well maintained (last commit was 2 days ago, http://comedi.org/git?p=comedi/comedi.git;a=summary). However, the driver for the DT9812 seems quite old:

Driver: dt9812
Description: Data Translation DT9812 USB module
Author: anders.blomdell@control.lth.se (Anders Blomdell)
Status: in development
Devices: [Data Translation] DT9812 (dt9812)
Updated: Sun Nov 20 20:18:34 EST 2005

This driver works, but bulk transfers not implemented. Might be a starting point
for someone else. I found out too late that USB has too high latencies (>1 ms)
for my needs.

### Setup at CBS - Knut presentation PC ###
*Welcome to the 90s*

- Windows XP Professional 2002 (SP2)
- Pentium 4 (3.00 GHz, 3 GB RAM)
- Matlab 7.5.0.342 (R2007b)
- Data Acquisition Toolbox 2.11
- Psychtoolbox 3.0.11
- DAQ Adaptor for Matlab
- Data Translation Open Layers (OEM)

### maybe useful code from Soyoung ###

```
#!matlab

%% dtol-----------------------------------------------------------------
rehash toolboxcache; % this is to update the toolboxcache, mainly this needs to be done if toolbox caching is enabled in the preferences (which speeds up matlab startup) and there were some changes
daqregister 'C:\Programme\Data Translation\DAQ Adaptor for MATLAB\dtol.dll' % this adds the DT library to matlab/daqtoolbox environment
openDAQ=daqfind; % returns all daq objects currently existing in the data acquisition
engine
for i=1:length(openDAQ),
  stop(openDAQ(i)); % this seems to stop a timer in the loaded objects, whatever that means
end
% from here we know everything :)
a1o=analogoutput('dtol')
addchannel(a1o,1)
set(a1o, 'samplerate',1000)
data(1:1000) = 0.01;
data = data';
% %% --------------------------------------------------------------------
```