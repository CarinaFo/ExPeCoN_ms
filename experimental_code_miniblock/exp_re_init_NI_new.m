function [aio_s,trig_s,ao] = exp_re_init_NI_new(exp_dir,thr_dir,p_data,ao_num,s_rate)
% [aio_s,trig_s,ao] = exp_re_init(exp_dir,p_data) runs initial procedures for 
% experiment:
%   - sets paths
%   - starts diary
%	- aio_setup
%
% Author:           Martin Grund
% Last update:      September 13, 2019

%%
% Make all assets available (e.g., Palamedes toolbox)
addpath(genpath([pwd, '/', exp_dir]))
addpath(genpath([pwd, '/', thr_dir]))
addpath(genpath([pwd, '/assets']))

%% Diary logfile   
diary([p_data.dir 'exp_' p_data.ID '_log.txt']);

%% Setup analog output (ao) and input (ai)
[aio_s,trig_s,ao] = aio_setup_NI_new(ao_num,s_rate);
