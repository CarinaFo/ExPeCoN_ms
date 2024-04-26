function [p_data,aio_s,trig_s,ao] = exp_init_NI_new(exp_dir,thr_dir,ao_num,s_rate)
% [p_data,aio_s,trig_s,ao] = exp_init(exp_dir) runs initial procedures for 
% experiment:
%   - sets paths
%   - starts diary
%   - participant_data
%	- aio_setup_NI_new
%
% Input variables_
% exp_dir       - directory name of experiment (e.g., respirationCA)
% thr_dir       - directory name of threshold assessment (e.g., thr1F or thr2F)
% ao_num        - number of analog output channels (number of stimulution sites + trigger channel)
% s_rate        - sampling rate for data acquisition (DAQ) card
%
% Output variables:
% p_data        - participant data (ID, gender, age)
% aio_s         - analog input/output session of data acquisition toolbox
% trig_s        - trigger session of data acquisition toolbox
% ao            - analog output channel object
%
% Author:           Martin Grund
% Last update:      July 22, 2021

%%
% Make all assets available (e.g., Palamedes toolbox)
addpath(genpath([pwd, '/', exp_dir]))
addpath(genpath([pwd, '/', thr_dir]))
addpath(genpath([pwd, '/assets']))

%% Particpant data
p_data = participant_data(['data/', exp_dir, '/ID']);

%% Diary logfile   
diary([p_data.dir 'exp_' p_data.ID '_log.txt']);

ver

%% Setup analog output (ao)
[aio_s,trig_s,ao] = aio_setup_NI_new(ao_num,s_rate);
