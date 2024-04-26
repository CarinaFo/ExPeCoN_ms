function [aio_s,trig_s,ao] = aio_setup_NI_new(ao_num,s_rate)
% s = aio_setup returns a trigger session (trig_s), an analog output 
% session (aio_s) and analog output object (ao) and input for the data 
% acquisition card from National Instruments (e.g., USB-6343)
% 
% Input
%   ao_num         - number of analog output channels
%   s_rate         - sampling rate (NI USB-6343: max. 500 kHz / ai_num)
% 
% Author:           Martin Grund
% Last update:      September 13, 2018


%% Stop any data acquisition objects

daqreset 
% resets Data Acquisition Toolbox and deletes all data acquisition session and device objects.

%% Trigger session

trig_s = daq.createSession('ni');
addDigitalChannel(trig_s, 'Dev1', 'Port0/Line0', 'OutputOnly');


%% Analog output session

aio_s = daq.createSession('ni');

% Settings
aio_s.Rate = s_rate;


%% Analog output setup

% Add channels
ao = addAnalogOutputChannel(aio_s,'Dev1',0:ao_num-1,'Voltage');


%% Trigger connection for analog output session

addTriggerConnection(aio_s, 'external', 'Dev1/PFI0', 'StartTrigger');
% Note: Output of 'Port0/Line0' is physically connected to PFI0

end